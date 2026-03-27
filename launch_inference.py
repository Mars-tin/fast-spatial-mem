#
#  Apache 2.0 License License
#  Copyright (c) 2026 Martin Ziqiao Ma
#
import os
import functools
import argparse
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from lpips import LPIPS

from fsm.data.data_mixer import get_dataset_inference
from fsm.model.metrics import (
    compute_ssim_from_imgs, 
    save_random_samples_with_psnr
)
from utils_ddp import (
    setup_ddp, 
    cleanup_ddp
)
from utils_train import (
    get_class_by_name,
    check_microarchitecture,
    get_ewc_config,
    discover_fast_blocks,
    init_ewc_buffers,
    maybe_reanchor,
    build_info
)

# Setup DDP
ctx = setup_ddp(backend="nccl")
device = ctx["device"]
dataloader_seed_generator = torch.Generator()
dataloader_seed_generator.manual_seed(ctx["seed"])

# Load config
parser = argparse.ArgumentParser(description="Override YAML values")
parser.add_argument("--config", "-c", type=str, required=True, help="Path to YAML configuration file")
parser.add_argument("--expname", "-n", type=str, default="fsm_inference")
parser.add_argument("--load_ckpt", type=str, default=None)
args = parser.parse_args()
config = OmegaConf.load(args.config)

is_inference = bool(getattr(config, "inference", False))
assert is_inference, "launch_inference.py should be run with inference mode enabled in config (inference: true)."

# Configure TF32
try:
    microarchitecture = check_microarchitecture(config)
except ValueError as e:
    exit(-1)
if getattr(config.training, "use_tf32", True):
    torch.backends.cuda.matmul.allow_tf32 = config.training.use_tf32
    torch.backends.cudnn.allow_tf32 = config.training.use_tf32

# Model
FSM = get_class_by_name(getattr(config.model, "class_name"))
model = FSM(config).to(device)

# Load checkpoint
now_iters = 0
output_dir = f"outputs/{args.expname}"
os.makedirs(output_dir, exist_ok=True)

ewc_buffers = None  # created lazily after model is built & (optionally) checkpoint restored

if os.path.isdir(output_dir):

    assert args.load_ckpt is not None, "In inference mode, --load_ckpt must be provided."
    checkpoint = torch.load(args.load_ckpt, map_location="cpu", weights_only=False)
    print(f"Loading checkpoint from {args.load_ckpt}...")

    msg = model.load_state_dict(checkpoint["model"], strict=True)
    print('Loading msg', msg)

    now_iters = checkpoint.get("now_iters", 0)
    if "ewc_buffers" in checkpoint:
        ewc_buffers = {}
        for k, v in checkpoint["ewc_buffers"].items():
            ewc_buffers[k] = {kk: vv.to(device=device) for kk, vv in v.items()}
        print("Restored EWC buffers from checkpoint.")

# Wrap DDP
if ctx["distributed"]:
    model = DDP(model, device_ids=[ctx["local_rank"]] if device.type == "cuda" else None)

# Activation checkpointing
if config.training.actckpt:

    def _check_fn(submodule) -> bool:
        from fsm.model.model_blocks import Block
        return isinstance(submodule, Block)
    
    torch._dynamo.config.optimize_ddp = False
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper as ptd_checkpoint_wrapper, 
        apply_activation_checkpointing
    )
    wrapper = functools.partial(ptd_checkpoint_wrapper, preserve_rng_state=False)
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=wrapper,
        check_fn=_check_fn,
    )

if config.training.torch_compile:
    print('Compiling Model..')
    model = torch.compile(model)

# Dataset and Dataloader
num_all_views = config.training.num_all_views
num_input_views = config.training.num_input_views
num_target_views = config.training.num_target_views

resolution = config.model.image_tokenizer.image_size
if isinstance(resolution, int):
    img_size = (resolution, resolution)
elif isinstance(resolution, (list, tuple)):
    img_size = tuple(resolution)

dataset_ratios = config.dataset.get("dataset_ratios", {})
dataset, sampler = get_dataset_inference(
    dataset_ratios,
    num_views=num_all_views,
    img_size=img_size,
    num_input_views=num_input_views,
    max_frame_time=config.model.get("max_frames", 256),
    scene_pose_normalize=config.dataset.get("scene_pose_normalize", True),
    window_size=config.dataset.get("window_size", 256),
    is_inference=True,
    sorted_indices=config.dataset.get("sort_by_time", True)
)

dataloader = DataLoader(
    dataset,
    batch_size=config.training.bs_per_gpu,
    shuffle=False,
    sampler=sampler,
    num_workers=config.training.num_workers,
    persistent_workers=True,
    pin_memory=config.training.get("pin_memory", True),
    drop_last=False,    # Mark as False for inference to get all samples!
    prefetch_factor=config.training.prefetch_factor,
    generator=dataloader_seed_generator,
)

# WandB setup
is_main = (dist.get_rank() == 0)
os.environ["WANDB_MODE"] = "disabled"

# EWC knobs from config.training
ewc_config = get_ewc_config(config)
ewc_enable = ewc_config["enable"]
ewc_lambda_prox = ewc_config["lambda_prox"]
ewc_alpha = ewc_config["alpha"]
ewc_mode = ewc_config["mode"]
ewc_anchor = ewc_config["anchor_policy"]
ewc_anchor_beta = ewc_config["anchor_beta"]
ewc_loss_weight = ewc_config["lambda_train"]

# Discover fast-weight blocks & init EWC buffers if needed (lazily)
fast_blocks = discover_fast_blocks(model)
if ewc_enable and (ewc_buffers is None):
    ewc_buffers = init_ewc_buffers(fast_blocks, device=device)
    
# store beta on blocks if ema policy used
anchor_mode = 1 if "streaming" in ewc_anchor else 0
if "ema" in ewc_anchor:
    anchor_mode *= 2
for _, blk in fast_blocks:
    setattr(blk, "anchor_mode", anchor_mode)
    if "ema" in ewc_anchor:
        setattr(blk, "anchor_beta", ewc_anchor_beta)

fsm_fast_blocks = discover_fast_blocks(model)

cum_total_imgs  = 0
cum_sum_psnr    = 0.0
cum_sum_lpips   = 0.0
cum_sum_ssim    = 0.0
cum_total_items = 0

print('Data loader length', len(dataloader))

# Re-anchor policy per epoch (-online): 
if ewc_enable and "online" in ewc_anchor:
    maybe_reanchor(fast_blocks, ewc_buffers, ewc_anchor)

if sampler is not None and hasattr(sampler, "set_epoch"):
    sampler.set_epoch(0)

for data_dict in dataloader:
    data_dict = {key: value.to(device) for key, value in data_dict.items() if isinstance(value, torch.Tensor)}
    input_data_dict = {key: value[:, :num_input_views] for key, value in data_dict.items()}
    target_data_dict = {key: value[:, -num_target_views:] for key, value in data_dict.items()}

    batch_size = next(iter(data_dict.values())).shape[0]  # batch size from any tensor

    # Build LaCT/EWC info for this batch (single, shared info; assumes uniform fast blocks)
    fsm_info_dict = build_info(
        fsm_fast_blocks, ewc_buffers, batch_size=batch_size,
        ewc_enable=ewc_enable, lambda_prox=ewc_lambda_prox, alpha=ewc_alpha, mode=ewc_mode
    )

    cum_total_items += batch_size

    model.eval()
    with torch.no_grad(), torch.autocast(dtype=torch.bfloat16, device_type="cuda", enabled=True):

        rendering, _ = model(input_data_dict, target_data_dict, info=fsm_info_dict, skip_loss=True)
        target = target_data_dict["image"]

        pred = rendering.flatten(0, 1) if rendering.ndim == 5 else rendering
        tgt = target.flatten(0, 1)     if target.ndim    == 5 else target

        # PSNR
        mse_per_image = F.mse_loss(pred, tgt, reduction='none')
        mse_per_image = mse_per_image.mean(dim=(1,2,3))
        eps = 1e-12
        psnr = -10.0 * torch.log10(mse_per_image.clamp_min(eps))
        
        # SSIM
        ssim = compute_ssim_from_imgs(rendering, target)

        # LPIPS
        lpips_loss_module = LPIPS(net="vgg").to(device).eval()
        lpips = lpips_loss_module(pred, tgt, normalize=True).mean()

        n_imgs = pred.shape[0]
        cum_total_imgs += n_imgs
        cum_sum_psnr   += float(psnr.sum().item())
        cum_sum_lpips  += float(lpips.item()) * n_imgs
        cum_sum_ssim   += float(ssim) * n_imgs

        if is_main:
            print(
                f"[EVAL] "
                f"Iter {now_iters:07d} | "
                f"PSNR {psnr.sum().item() / n_imgs:.2f} | "
                f"LPIPS {lpips.item():.4f} | "
                f"SSIM {ssim:.4f}"
            )

            # Save & (optionally) log N random samples
            try:
                _ = save_random_samples_with_psnr(
                    rendering=rendering, 
                    target=target,
                    now_iters=now_iters, 
                    output_dir=output_dir+'_eval',
                    sample_n=3, 
                    wandb_run=False, 
                    title_prefix="render_vs_gt_with_psnr"
                )
            except Exception as e:
                print(f"[warn] sample logging failed at iter {now_iters}: {e}")

    now_iters += 1
    continue

# Finalize evaluation metrics across all processes
if torch.distributed.is_available() and torch.distributed.is_initialized():
    _tot  = torch.tensor([cum_total_imgs], dtype=torch.float64, device=device)
    _sums = torch.tensor([cum_sum_psnr, cum_sum_lpips, cum_sum_ssim],
                            dtype=torch.float64, device=device)
    t = torch.tensor([cum_total_items], device=device, dtype=torch.long)
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(_tot,  op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(_sums, op=torch.distributed.ReduceOp.SUM)
    if torch.distributed.get_rank() == 0 and _tot.item() > 0:
        print(f"[EVAL][FINAL] avg PSNR {_sums[0].item()/_tot.item():.3f} | "
                f"avg LPIPS {_sums[1].item()/_tot.item():.4f} | "
                f"avg SSIM {_sums[2].item()/_tot.item():.4f}")
        print("global_items", int(t.item()), "dataset_len", len(dataset))
else:
    if cum_total_imgs > 0:
        print(f"[EVAL][FINAL] avg PSNR {cum_sum_psnr/cum_total_imgs:.3f} | "
                f"avg LPIPS {cum_sum_lpips/cum_total_imgs:.4f} | "
                f"avg SSIM {cum_sum_ssim/cum_total_imgs:.4f}")

cleanup_ddp()
