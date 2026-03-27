#
#  Apache 2.0 License License
#  Copyright (c) 2026 Martin Ziqiao Ma
#
import os
import functools
import argparse
from pathlib import Path
from omegaconf import OmegaConf

import math
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import wandb

from fsm.data.data_mixer import get_dataset
from fsm.model.losses.ewc_loss import ewc_training_loss
from fsm.model.metrics import (
    log_rendering_metrics, 
    save_random_samples_with_psnr
)
from utils_ddp import (
    setup_ddp, 
    cleanup_ddp, 
    apply_linear_scaling
)
from utils_train import (
    get_class_by_name,
    check_microarchitecture,
    get_optimizer,
    get_lr_scheduler,
    get_ewc_config,
    save_checkpoint,
    discover_fast_blocks,
    init_ewc_buffers,
    maybe_reanchor,
    build_info
)

def every(n, step, also_first=10):
    return (step % n == 0) or (step <= also_first)

# Setup DDP
ctx = setup_ddp(backend="nccl")
device = ctx["device"]
dataloader_seed_generator = torch.Generator()
dataloader_seed_generator.manual_seed(ctx["seed"])

# Load config
parser = argparse.ArgumentParser(description="Override YAML values")
parser.add_argument("--config", "-c", type=str, required=True, help="Path to YAML configuration file")
parser.add_argument("--expname", "-n", type=str, default="fsm_training")
parser.add_argument("--wandb_name", type=str, default=None)
parser.add_argument("--load_ckpt", type=str, default=None)
parser.add_argument("--remove_linear", type=bool, default=False)
args = parser.parse_args()
config = OmegaConf.load(args.config)

is_inference = bool(getattr(config, "inference", False))
assert not is_inference, "launch_training.py should be run with training mode enabled in config (inference: false)."

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

# Optimizers and schedulers
scale = apply_linear_scaling(config, ctx)
optimizer = get_optimizer(
    model,
    weight_decay=config.training.weight_decay,
    learning_rate=config.training.lr,
    betas=config.training.betas
)
lr_scheduler = get_lr_scheduler(
    optimizer,
    warm_up_steps=config.training.warmup_steps,
    total_train_steps=config.training.steps,
    scheduler_type="cosine"
)

# AMP Grad Scaler
if microarchitecture == "amp":
    enable_grad_scaler = config.training.use_amp and config.training.amp_dtype == "fp16"
    scaler = torch.cuda.amp.GradScaler(enabled=enable_grad_scaler)
    amp_dtype_mapping = {"fp16": torch.float16, "bf16": torch.bfloat16}
    # [TODO]: right now we are running on H100 with bf16, will implement A100 supports.
    raise NotImplemented

# Load checkpoint
now_iters = 0
output_dir = f"outputs/{args.expname}"
os.makedirs(output_dir, exist_ok=True)

ewc_buffers = None  # created lazily after model is built & (optionally) checkpoint restored

if os.path.isdir(output_dir):

    checkpoints = [f for f in os.listdir(output_dir) if f.startswith("model_") and f.endswith(".pth")]
    
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("_")[1].split(".")[0]))
        checkpoint_path = os.path.join(output_dir, latest_checkpoint)
        print(f"Loading checkpoint from {checkpoint_path}...")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])

        if not getattr(config.training, "reset_optim", False):
            print(f"Restoring optimizer and lr_scheduler states...")
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            now_iters = checkpoint.get("now_iters", 0)

        # Try to restore ewc_buffers if present (backward compatible if not)
        if "ewc_buffers" in checkpoint:
            ewc_buffers = {}
            for k, v in checkpoint["ewc_buffers"].items():
                ewc_buffers[k] = {kk: vv.to(device=device) for kk, vv in v.items()}
            print("Restored EWC buffers from checkpoint.")

    elif args.load_ckpt is not None:
        checkpoint_path = args.load_ckpt
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        #remove input_linear weights if present
        if args.remove_linear:
            print('Removing input linear of 3d model')
            checkpoint["model"] = {k: v for k, v in checkpoint["model"].items() if not k.startswith("input_linear.")}
        msg = model.load_state_dict(checkpoint["model"], strict=True)
        print('Loaded checkpoint with message:', msg)

print(f"Starting from iter {now_iters}.")

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

# Model Compilation
if config.training.torch_compile:
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
dataset, sampler = get_dataset(
    dataset_ratios,
    num_views=num_all_views,
    img_size=img_size,
    num_input_views=num_input_views,
    max_frame_time=config.model.get("max_frames", 256),
    scene_pose_normalize=config.dataset.get("scene_pose_normalize", True),
    window_size=config.dataset.get("window_size", 256),
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
    drop_last=True,
    prefetch_factor=config.training.prefetch_factor,
    generator=dataloader_seed_generator
)

# WandB Setup
run_name = (
    f"bs{config.training.bs_per_gpu}_"
    f"in{num_input_views}_"
    f"out{num_target_views}_"
    f"all{num_all_views}_"
    f"lr{config.training.lr:g}"
)
if getattr(config.wandb, "wandb_name", None):
    run_name = config.wandb.wandb_name
if getattr(args, "wandb_name", None):
    run_name = args.wandb_name

is_main = (dist.get_rank() == 0)
if is_main:
    wandb_run = wandb.init(
        entity=config.wandb.wandb_entity,
        project=config.wandb.wandb_project,
        name=run_name,
        mode=getattr(config.wandb, "wandb_mode", "online"),
        config=OmegaConf.to_container(config, resolve=True),
        dir=output_dir,
    )
    print(model)
    print(optimizer)
    print(lr_scheduler)
    print(f"Start training from iter {now_iters}...")
else:
    os.environ["WANDB_MODE"] = "disabled"
    wandb_run = None

# EWC knobs from config.training
ewc_config = get_ewc_config(config)
ewc_enable = ewc_config["enable"]
ewc_lambda_prox = ewc_config["lambda_prox"]
ewc_alpha = ewc_config["alpha"]
ewc_mode = ewc_config["mode"]
ewc_anchor = ewc_config["anchor_policy"]
ewc_anchor_beta = ewc_config["anchor_beta"]
ewc_loss_weight = ewc_config["lambda_train"]

# Training loop
remaining_steps = config.training.steps - now_iters

# Discover fast-weight blocks & init EWC buffers if needed (lazily)
fast_blocks = discover_fast_blocks(model)
if ewc_enable and (ewc_buffers is None):
    ewc_buffers = init_ewc_buffers(fast_blocks, device=device)
    
# Store beta on blocks if ema policy used
# [TODO] make this an enum for readability, for now 摆了...
anchor_mode = 1 if "streaming" in ewc_anchor else 0
if "ema" in ewc_anchor:
    anchor_mode *= 2
for _, blk in fast_blocks:
    setattr(blk, "anchor_mode", anchor_mode)
    if "ema" in ewc_anchor:
        setattr(blk, "anchor_beta", ewc_anchor_beta)

fsm_fast_blocks = discover_fast_blocks(model)

print(f"Starting training for {remaining_steps} steps...")
for epoch in range((remaining_steps - 1) // len(dataloader) + 1):
    dataloader.sampler.set_epoch(epoch)

    # Re-anchor policy per epoch (-online): 
    if ewc_enable and "online" in ewc_anchor:
        maybe_reanchor(fast_blocks, ewc_buffers, ewc_anchor)

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

        if isinstance(model, DDP):
            model.module.set_curriculum(now_iters)
        else:
            model.set_curriculum(now_iters)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(dtype=torch.bfloat16, device_type="cuda", enabled=True):

            # Model forward
            rendering, losses = model(
                input_data_dict, 
                target_data_dict, 
                info=fsm_info_dict
            )
            loss = losses.loss

            # Optional training-time EWC loss
            if ewc_enable and ewc_loss_weight > 0.0 and fast_blocks:
                ewc_loss_train = ewc_training_loss(fast_blocks, ewc_buffers)
                loss = loss + ewc_loss_weight * ewc_loss_train
            else:
                ewc_loss_train = torch.tensor(0.0, device=device)

        loss.backward()

        # Gradient safeguard
        skip_optimizer_step = False
        if now_iters <= config.training.get("grad_clip_warmup_steps", 1000):
            global_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))
        else:
            global_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.get("grad_clip_norm", 1.0))
            if not math.isfinite(global_grad_norm):
                skip_optimizer_step = True
            elif global_grad_norm > config.training.get("grad_threshold", 4.0):
                skip_optimizer_step = True

        if not skip_optimizer_step:
            optimizer.step()

        lr_scheduler.step()
        now_iters += 1

        # Logging and visualization
        if is_main and every(config.training.log_every, now_iters):

            model.eval()
            with torch.no_grad():

                # WandB log
                if wandb_run:
                    log_rendering_metrics(
                        now_iters, 
                        epoch, 
                        loss, 
                        losses.l2_loss, 
                        losses.psnr, 
                        losses.perceptual_loss, 
                        losses.lpips_loss, 
                        losses.ssim_loss,
                        losses.pixelalign_loss,
                        losses.pointsdist_loss,
                        grad_norm=global_grad_norm, 
                        optimizer=optimizer,
                        gaussians_usage=None
                    )

                # Console log
                parts = [f"Iter {now_iters:07d}"]
                if losses.loss is not None:
                    parts.append(f"Loss {losses.loss.item():.4f}")
                if losses.psnr is not None:
                    parts.append(f"PSNR {losses.psnr:.2f}")
                if losses.lpips_loss is not None:
                    parts.append(f"LPIPS {losses.lpips_loss.item():.4f}")
                if losses.l2_loss is not None:
                    parts.append(f"L2 {losses.l2_loss.item():.4f}")
                if losses.perceptual_loss is not None:
                    parts.append(f"Perceptual {losses.perceptual_loss.item():.4f}")
                if losses.ssim_loss is not None:
                    parts.append(f"SSIM {losses.ssim_loss.item():.4f}")
                if losses.pixelalign_loss is not None:
                    parts.append(f"PixelAlign {losses.pixelalign_loss.item():.4f}")
                if losses.pointsdist_loss is not None:
                    parts.append(f"PointsDist {losses.pointsdist_loss.item():.4f}")
                print(" | ".join(parts) + " |")

                # Save & (optionally) log N=3 random samples
                try:
                    if every(config.training.visualize_every, now_iters):
                        _ = save_random_samples_with_psnr(
                            rendering=rendering,
                            target=target_data_dict["image"],
                            now_iters=now_iters,
                            output_dir=output_dir,
                            sample_n=3,
                            wandb_run=False,
                            title_prefix="render_vs_gt_with_psnr"
                        )
                except Exception as e:
                    print(f"[warn] sample logging failed at iter {now_iters}: {e}")

            model.train()

        # Checkpointing
        if is_main and every(config.training.save_every, now_iters, also_first=-1):
            save_checkpoint(
                Path(output_dir) / f"model_{now_iters:07d}.pth",
                model, optimizer, lr_scheduler, now_iters, epoch, ewc_buffers
            )

        if now_iters == config.training.steps:
            break

if is_main and wandb_run:
    wandb_run.finish()

cleanup_ddp()
