#
# This file is derived from code released under the MIT License.
#
# Original copyright:
# Copyright (c) 2025 Tianyuan Zhang, Hao Tan
#
# Modifications copyright:
# Copyright (c) 2026 Martin Ziqiao Ma
#
# This file is distributed under the licensing terms provided in the
# repository LICENSE and NOTICE files.
#
import inspect
import torch
from pathlib import Path
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

def get_class_by_name(name: str):
    """Import and return a class from its fully qualified name.

    Args:
        name (str): Fully qualified class name, e.g. "package.module.ClassName".

    Returns:
        type: The class object resolved from the given name.
    """
    module_name, class_name = name.rsplit(".", 1)
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


def check_microarchitecture(config):
    """
    Ensures at least one of config.training.use_hop or config.training.use_amp is True,
    but not both.

    Raises:
        ValueError: if both are False or both are True.
    """
    use_hop = getattr(config.training, "use_hop", False)
    use_amp = getattr(config.training, "use_amp", False)

    # At least one must be True
    if not (use_hop or use_amp):
        raise ValueError("Either config.training.use_hop or config.training.use_amp must be True.")

    # But not both
    if use_hop and use_amp:
        raise ValueError("Both config.training.use_hop and config.training.use_amp cannot be True at the same time.")

    print(f"[Config] Using {'HOP' if use_hop else 'AMP'} mode.")
    return "hop" if use_hop else "amp"


def get_optimizer(model, weight_decay, learning_rate, betas):
    """Create an AdamW optimizer with standard decay and no-decay parameter groups.

    Parameters that require gradients are split by tensor dimensionality:
    tensors with dimension >= 2 receive weight decay, while tensors with
    dimension < 2 do not. When supported and applicable, the fused AdamW
    implementation is used on CUDA.

    Args:
        model: Model whose parameters will be optimized.
        weight_decay: Weight decay applied to decay parameter groups.
        learning_rate: Optimizer learning rate.
        betas: AdamW beta coefficients.

    Returns:
        torch.optim.Optimizer: Configured AdamW optimizer.
    """
    params = [p for _, p in model.named_parameters() if p.requires_grad]
    decay_params = [p for p in params if p.dim() >= 2]
    nodecay_params = [p for p in params if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    # Create AdamW optimizer and use the fused version if it is available
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and next(model.parameters()).is_cuda

    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(
        optim_groups, lr=learning_rate, betas=betas, **extra_args
    )

    return optimizer


def get_lr_scheduler(
    optimizer, total_train_steps, warm_up_steps, scheduler_type="cosine"
):
    """Create a learning-rate scheduler with warmup.

    Supports linear, cosine, and constant schedules, all with warmup. The
    scheduler is configured using the total number of training steps and the
    number of warmup steps.

    Args:
        optimizer: Optimizer whose learning rate will be scheduled.
        total_train_steps: Total number of training steps.
        warm_up_steps: Number of warmup steps.
        scheduler_type: Scheduler type. Must be one of ``"linear"``,
            ``"cosine"``, or ``"constant"``.

    Returns:
        Learning rate scheduler instance.

    Raises:
        ValueError: If ``scheduler_type`` is not supported.
    """
    if scheduler_type == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warm_up_steps,
            num_training_steps=total_train_steps,
        )
    elif scheduler_type == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warm_up_steps,
            num_training_steps=total_train_steps,
        )
    elif scheduler_type == "constant":
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warm_up_steps,
        )
    else:
        raise ValueError(f"Not support LR scheduler type {scheduler_type}.")

    return lr_scheduler


def remove_module_prefix(state_dict):
    """Remove common wrapper prefixes from state-dict parameter names.

    This strips prefixes introduced by distributed or compiled wrappers,
    including ``"_checkpoint_wrapped_module."``, ``"_orig_mod."``, and any
    repeated leading ``"module."`` segments.

    Args:
        state_dict: State dictionary mapping parameter names to values.

    Returns:
        dict: New state dictionary with normalized parameter names.
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        key = key.replace("_checkpoint_wrapped_module.", "")
        key = key.replace("_orig_mod.", "")
        while key.startswith("module."):
            key = key[len("module."):]
        new_state_dict[key] = value
    return new_state_dict


def save_checkpoint(
    path,
    model,
    optimizer,
    lr_scheduler,
    now_iters,
    epoch,
    ewc_buffers=None,
):
    """Save a training checkpoint to disk.

    The checkpoint includes the model state, optional optimizer and scheduler
    states, the current iteration and epoch, and optional EWC buffers. EWC
    buffers are moved to CPU before saving to keep checkpoints device-agnostic
    and smaller.

    Args:
        path: Output checkpoint path.
        model: Model whose parameters will be saved.
        optimizer: Optional optimizer whose state will be saved.
        lr_scheduler: Optional learning-rate scheduler whose state will be saved.
        now_iters: Current training iteration.
        epoch: Current epoch.
        ewc_buffers: Optional EWC buffer dictionary to include in the checkpoint.

    Returns:
        None.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "model": remove_module_prefix(model.state_dict()),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
        "now_iters": int(now_iters),
        "epoch": int(epoch),
        "format_version": 2,  # bump for clarity
    }

    if ewc_buffers is not None:
        # move to CPU to keep checkpoints smaller & device-agnostic
        cpu_buf = {}
        for name, buf_dict in ewc_buffers.items():
            cpu_buf[name] = {k: v.detach().to("cpu") for k, v in buf_dict.items()}
        state["ewc_buffers"] = cpu_buf

    torch.save(state, str(path))


def get_ewc_config(config):
    """Extract EWC-related training settings from the config.

    Reads EWC configuration fields from ``config.training`` and returns them in
    a normalized dictionary with explicit Python scalar types.

    Args:
        config: Config object expected to contain a ``training`` section.

    Returns:
        dict: Dictionary of EWC configuration values, including enable flag,
        regularization strengths, update mode, and anchor policy settings.

    Raises:
        ValueError: If ``config.training`` is missing.
    """
    training = getattr(config, "training", None)
    if training is None:
        raise ValueError("config.training not found")

    ewc_cfg = {
        "enable":        bool(getattr(training, "ewc_enable", False)),
        "lambda_prox":   float(getattr(training, "ewc_lambda_prox", 0.0)),
        "alpha":         float(getattr(training, "ewc_alpha", 0.95)),
        "mode":          str(getattr(training, "ewc_mode", "sq")),
        "anchor_policy": str(getattr(training, "ewc_anchor_policy", "global")),  # "global", "streaming", "ema"
        "anchor_beta":   float(getattr(training, "ewc_anchor_beta", 0.0)),
        "lambda_train":  float(getattr(training, "ewc_lambda_train", 0.0)),
    }
    return ewc_cfg


def _is_fast_weight_block(m):
    """Return whether a module matches the expected fast-weight block pattern.

    This intentionally avoids importing the block class directly and instead
    checks for the required attribute pattern together with the class name.

    Args:
        m: Module to inspect.

    Returns:
        bool: True if the module matches the expected fast-weight block
        signature, otherwise False.
    """
    return (
        hasattr(m, "w0") and hasattr(m, "w1") and hasattr(m, "w2")
        and hasattr(m, "num_heads")
        and type(m).__name__ == "FastWeightGluMLPMultihead"
    )


def discover_fast_blocks(model):
    """Discover all fast-weight blocks in a model.

    Args:
        model: Model to traverse via ``named_modules()``.

    Returns:
        list: List of ``(name, module)`` pairs for all discovered fast-weight
        blocks.
    """
    blocks = []
    for name, m in model.named_modules():
        if _is_fast_weight_block(m):
            blocks.append((name, m))
    return blocks


def discover_fast_blocks_by_name(model, module_name):
    """Discover fast-weight blocks whose module names contain a substring.

    Args:
        model: Model to traverse via ``named_modules()``.
        module_name: Substring used to filter discovered module names.

    Returns:
        list: List of ``(name, module)`` pairs for matching fast-weight blocks.
    """
    blocks = []
    for name, m in model.named_modules():
        if _is_fast_weight_block(m) and (module_name in name):
            blocks.append((name, m))
    return blocks


def init_ewc_buffers(blocks, device):
    """Create per-block anchor and Fisher buffers (per-head shapes)."""
    ewc_buffers = {}
    for name, blk in blocks:
        w0_star = blk.w0.detach().to(device=device, dtype=torch.float32).clone()
        w1_star = blk.w1.detach().to(device=device, dtype=torch.float32).clone()
        w2_star = blk.w2.detach().to(device=device, dtype=torch.float32).clone()
        F0 = torch.full_like(w0_star, 1e-6, dtype=torch.float32, device=device)
        F1 = torch.full_like(w1_star, 1e-6, dtype=torch.float32, device=device)
        F2 = torch.full_like(w2_star, 1e-6, dtype=torch.float32, device=device)
        ewc_buffers[name] = {
            "w0_star": w0_star, "w1_star": w1_star, "w2_star": w2_star,
            "F0": F0, "F1": F1, "F2": F2,
        }
    return ewc_buffers


def maybe_reanchor(blocks, ewc_buffers, policy: str):
    """Apply global/streaming policy. Global = no-op; Streaming = reanchor to current weights."""
    if policy not in ("streaming", "global", "ema"):
        return  # unknown: ignore
    if policy == "global":
        return
    for name, blk in blocks:
        buf = ewc_buffers[name]
        if policy == "streaming":
            buf["w0_star"].copy_(blk.w0.detach())
            buf["w1_star"].copy_(blk.w1.detach())
            buf["w2_star"].copy_(blk.w2.detach())
        elif policy == "ema":
            # if ema is desired but no beta specified, fall back to mild EMA=0.95
            beta = getattr(blk, "ewc_anchor_beta", 0.95)
            buf["w0_star"].mul_(beta).add_(blk.w0.detach(), alpha=(1.0 - beta))
            buf["w1_star"].mul_(beta).add_(blk.w1.detach(), alpha=(1.0 - beta))
            buf["w2_star"].mul_(beta).add_(blk.w2.detach(), alpha=(1.0 - beta))


def _expand_for_batch(t: torch.Tensor, B: int):
    """Repeat per-head tensor [H, ...] to [B*H, ...] along dim=0."""
    return torch.repeat_interleave(t, repeats=B, dim=0)


def build_info(
    blocks, ewc_buffers, batch_size: int, ewc_enable: bool, lambda_prox: float, alpha: float, mode: str
):
    """Return a dict of tensors/scalars to be read by fast-weight blocks.
    IMPORTANT: This assumes that all fast-weight blocks share the same head layout (common in practice).
    If your architecture differs per block, consider per-block info wiring in the model instead.
    """
    info = {}

    # TTT operator order must already be set by the model; we do not construct it here.
    # The model should merge our EWC keys into each layer's info internally if it has multiple blocks.
    info["ewc_enabled"] = bool(ewc_enable)
    info["lambda_ewc"] = float(lambda_prox)
    info["fisher_alpha"] = float(alpha)

    # [TODO] make this an enum for readability, for now 摆了...
    if mode == "abs":
        info["fisher_mode"] = 0
    elif mode == "sq":
        info["fisher_mode"] = 1
    elif mode == "si":
        info["fisher_mode"] = 2
    else:
        info["fisher_mode"] = -1
    
    if not ewc_enable:
        return info  # keep info minimal

    if not blocks:
        return info

    # Use the first block's shapes to expand anchors/Fishers for the current batch.
    # This works if all blocks are shape-identical (typical). If not, consider moving expansion into the model layer.
    name0, blk0 = blocks[0]
    buf0 = ewc_buffers[name0]
    w0_star = _expand_for_batch(buf0["w0_star"], batch_size).to(device=blk0.w0.device, dtype=blk0.w0.dtype)
    w1_star = _expand_for_batch(buf0["w1_star"], batch_size).to(device=blk0.w1.device, dtype=blk0.w1.dtype)
    w2_star = _expand_for_batch(buf0["w2_star"], batch_size).to(device=blk0.w2.device, dtype=blk0.w2.dtype)
    F0 = _expand_for_batch(buf0["F0"], batch_size).to(device=blk0.w0.device, dtype=torch.float32)
    F1 = _expand_for_batch(buf0["F1"], batch_size).to(device=blk0.w1.device, dtype=torch.float32)
    F2 = _expand_for_batch(buf0["F2"], batch_size).to(device=blk0.w2.device, dtype=torch.float32)

    info.update({
        "w0_star": w0_star, "w1_star": w1_star, "w2_star": w2_star,
        "F0": F0, "F1": F1, "F2": F2,
    })
    return info

def remap_ewc_buffers(ewc_buffers):
    """Remap EWC buffer keys to match the unwrapped model structure.

    The checkpoint was saved from a DDP + activation-checkpointed model, which may
    prepend module name wrappers such as ``"module."`` and
    ``"_checkpoint_wrapped_module."``. Strip both so the buffer keys match the
    base model.
    """
    remapped = {}
    for k, v in ewc_buffers.items():
        new_k = k[len("module."):] if k.startswith("module.") else k
        new_k = new_k.replace("._checkpoint_wrapped_module.", ".")
        remapped[new_k] = v
    return remapped
