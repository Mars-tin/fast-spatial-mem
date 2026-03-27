#
# This file is derived from code released under the Apache 2.0 License.
#
# Original copyright:
# Copyright (c) 2025 Ziwen Chen
#
# Modifications copyright:
# Copyright (c) 2026 Martin Ziqiao Ma
#
# This file is distributed under the licensing terms provided in the
# repository LICENSE and NOTICE files.
#
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
import wandb
from skimage.metrics import structural_similarity as ssim_fn
import json
from pathlib import Path


def save_random_samples_with_psnr(
    rendering: torch.Tensor,
    target: torch.Tensor,
    now_iters: int,
    output_dir: str,
    sample_n: int = 2,
    wandb_run=None,
    title_prefix: str = "render_vs_gt_with_psnr"
):
    """
    Randomly sample N images from rendering/target, and save ONE concatenated image:
      [pred | gt | PSNR text] per row, stacked for N rows.

    Output path: Path(output_dir) / f"{now_iters}" / (title_prefix + ".png")
    """
    pred = _flatten_to_nchw(rendering)
    gt   = _flatten_to_nchw(target)

    assert pred.shape == gt.shape, f"pred {tuple(pred.shape)} vs gt {tuple(gt.shape)} mismatch"
    total = pred.shape[0]
    n = min(sample_n, total)

    # random, without replacement
    idx = torch.randperm(total, device=pred.device)[:n]
    pred_s = pred.index_select(0, idx)
    gt_s   = gt.index_select(0, idx)

    psnrs = _per_sample_psnr(pred_s, gt_s)  # (n,)

    # font
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    rows = []
    for i in range(n):
        pimg = _to_uint8_img(pred_s[i])
        gimg = _to_uint8_img(gt_s[i])

        h = pimg.height
        text = f"PSNR: {psnrs[i].item():.2f} dB"

        # measure text and build panel with padding
        pad = 12
        dummy_panel = Image.new("RGB", (1, 1), (255, 255, 255))
        draw_dummy = ImageDraw.Draw(dummy_panel)
        tw, th = _measure_text(draw_dummy, text, font)
        text_w = max(190, tw + 2 * pad)  # keep your old min width but adapt as needed

        text_panel = Image.new("RGB", (text_w, h), (255, 255, 255))
        draw = ImageDraw.Draw(text_panel)
        x = (text_w - tw) // 2
        y = (h - th) // 2
        draw.text((x, y), text, fill=(0, 0, 0), font=font)

        row = Image.new("RGB", (pimg.width + gimg.width + text_w, h), (255, 255, 255))
        x = 0
        row.paste(pimg, (x, 0)); x += pimg.width
        row.paste(gimg, (x, 0)); x += gimg.width
        row.paste(text_panel, (x, 0))
        rows.append(row)

    total_h = sum(r.height for r in rows)
    total_w = max(r.width for r in rows)
    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    y = 0
    for r in rows:
        canvas.paste(r, (0, y)); y += r.height

    out_dir = Path(output_dir) / f"{now_iters}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{title_prefix}.png"
    canvas.save(out_path.as_posix())

    if wandb_run or wandb_run is not None:
        import wandb
        wandb_run.log({title_prefix: wandb.Image(canvas)}, step=now_iters)

    return out_path.as_posix()


def save_all_samples_with_psnr(
    rendering: torch.Tensor,
    target: torch.Tensor,
    now_iters: int,
    output_dir: str,
    json_ids=None,
    frame_times=None,
    use_batch_index_as_id: bool = True,
):
    """
    Save *all* samples in the batch.

    For each logical "sample" (batch index), create a folder under:
        Path(output_dir) / f"{now_iters:07d}" / f"{sample_id}"

    Inside each folder, save:
        - pred_{view_idx}.png   (predictions)
        - gt_{view_idx}.png     (ground truth)
        - psnr.json             (includes json_id, frame_time, per-view PSNR)

    Assumes rendering/target shapes:
        - (B, V, C, H, W)  or
        - (B, C, H, W)     (treated as V = 1)
    """

    # Ensure 5D: (B, V, C, H, W)
    if rendering.ndim == 4:
        rendering = rendering.unsqueeze(1)  # (B, 1, C, H, W)
        target    = target.unsqueeze(1)
    assert rendering.ndim == 5, f"Expected 4D/5D, got {rendering.shape}"
    assert target.shape == rendering.shape, f"rendering {rendering.shape} vs target {target.shape}"

    B, V, C, H, W = rendering.shape

    # Flatten to NCHW for PSNR computation
    pred_flat = rendering.reshape(B * V, C, H, W)
    gt_flat   = target.reshape(B * V, C, H, W)

    # Per-image PSNR (N = B * V)
    psnr_flat = _per_sample_psnr(pred_flat, gt_flat)  # (N,)

    # Root directory for this iteration
    root_dir = Path(output_dir) / f"{now_iters:07d}"
    root_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Normalize json_ids ----------
    norm_json_ids = None
    if json_ids is not None:
        if isinstance(json_ids, torch.Tensor):
            if json_ids.ndim == 1:
                norm_json_ids = [str(json_ids[i].item()) for i in range(B)]
            else:
                norm_json_ids = [str(list(json_ids[i].cpu().numpy())) for i in range(B)]
        elif isinstance(json_ids, (list, tuple)):
            norm_json_ids = [str(j) for j in json_ids]
        else:
            # broadcast scalar-ish
            norm_json_ids = [str(json_ids)] * B

    # ---------- Normalize frame_times ----------
    # Expect shape (B, V) or (B,) tensor, or list-of-lists
    norm_frame_times = None
    if frame_times is not None:
        if isinstance(frame_times, torch.Tensor):
            if frame_times.ndim == 1:
                # One frame_time per sample
                norm_frame_times = [[float(frame_times[i].item())] * V for i in range(B)]
            elif frame_times.ndim == 2:
                # (B, V)
                norm_frame_times = [
                    [float(x) for x in frame_times[i].tolist()]
                    for i in range(B)
                ]
            else:
                raise ValueError(f"Unexpected frame_times shape: {frame_times.shape}")
        elif isinstance(frame_times, (list, tuple)):
            # Try to interpret as list-of-lists or list of scalars
            if len(frame_times) == B and isinstance(frame_times[0], (list, tuple)):
                norm_frame_times = [
                    list(map(float, ft_list)) for ft_list in frame_times
                ]
            elif len(frame_times) == B:
                norm_frame_times = [
                    [float(ft)] * V for ft in frame_times
                ]
            else:
                # broadcast single value to all samples & views
                norm_frame_times = [
                    [float(frame_times)] * V for _ in range(B)
                ]
        else:
            # single scalar → broadcast
            norm_frame_times = [
                [float(frame_times)] * V for _ in range(B)
            ]

    # Loop per batch sample
    for b in range(B):
        # Folder name (still using batch index)
        if use_batch_index_as_id:
            sample_id = f"{b:06d}"
        else:
            sample_id = f"{b:06d}"

        sample_dir = root_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Base info for this sample
        sample_json_id = norm_json_ids[b] if norm_json_ids is not None else None
        sample_frame_times = norm_frame_times[b] if norm_frame_times is not None else None  # list length V or None

        psnr_dict = {
            "json_id": sample_json_id,
            "sample_id": sample_id,
            "frame_time": sample_frame_times,  # list of frame_time per view
            "views": {},                      # per-view PSNR + frame_time
        }

        for v in range(V):
            idx = b * V + v

            pimg = _to_uint8_img(pred_flat[idx])  # PIL image
            gimg = _to_uint8_img(gt_flat[idx])

            pred_name = f"pred_{v:03d}.png"
            gt_name   = f"gt_{v:03d}.png"

            pimg.save(sample_dir / pred_name)
            gimg.save(sample_dir / gt_name)

            psnr_val = float(psnr_flat[idx].item())
            ft_val = None
            if sample_frame_times is not None and v < len(sample_frame_times):
                ft_val = sample_frame_times[v]

            psnr_dict["views"][pred_name] = {
                "psnr": psnr_val,
                "frame_time": ft_val,
                "gt": gt_name,
            }

        # Write PSNRs as JSON
        with open(sample_dir / "psnr.json", "w") as f:
            json.dump(psnr_dict, f, indent=2)

    return root_dir.as_posix()


def compute_ssim_from_imgs(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Compute mean SSIM between predicted and ground-truth images.

    Args:
        pred: (N, C, H, W) or (N, V, C, H, W), values in [0, 1].
        gt:   same shape as pred.

    Returns:
        scalar float: mean SSIM across all samples.
    """
    with torch.no_grad():
        if pred.ndim == 5:
            pred = pred.flatten(0, 1)
            gt = gt.flatten(0, 1)
        pred_np = pred.clamp(0, 1).to(torch.float32).detach().cpu().numpy()
        gt_np   = gt.clamp(0, 1).to(torch.float32).detach().cpu().numpy()
        vals = [
            ssim_fn(
                g, p,
                win_size=11,
                gaussian_weights=True,
                channel_axis=0,
                data_range=1.0,
            )
            for g, p in zip(gt_np, pred_np)
        ]
        return float(sum(vals) / len(vals)) if vals else float("nan")
    

def log_rendering_metrics(
    now_iters,
    epoch,
    loss,
    l2_loss,
    psnr,
    perceptual_loss=None,
    lpips_loss=None,
    ssim_loss=None,
    pixelalign_loss=None,
    pointsdist_loss=None,
    grad_norm=None,
    optimizer=None,
    gaussians_usage=None,
):
    """Log rendering metrics to Weights & Biases.

    Args:
        now_iters: Current global iteration used as the logging step.
        epoch: Current epoch number.
        loss: Total loss tensor.
        l2_loss: L2 reconstruction loss tensor.
        psnr: PSNR value.
        perceptual_loss: Optional perceptual loss value.
        lpips_loss: Optional LPIPS loss value.
        ssim_loss: Optional SSIM value.
        pixelalign_loss: Optional pixel-alignment loss value.
        pointsdist_loss: Optional points-distance loss value.
        grad_norm: Optional gradient norm value.
        optimizer: Optional optimizer used to read the learning rate.
        gaussians_usage: Optional Gaussian usage statistic.
    """
    log_dict = {
        "epoch": int(epoch),
        "loss": float(loss.item()),
        "l2_loss": float(l2_loss.item()),
        "psnr": psnr
    }

    if lpips_loss is not None:
        log_dict["lpips_loss"] = float(lpips_loss.item())

    if perceptual_loss is not None:
        log_dict["perceptual_loss"] = float(perceptual_loss.item())

    if ssim_loss is not None:
        log_dict["ssim_loss"] = float(ssim_loss.item())
    
    if pixelalign_loss is not None:
        log_dict["pixelalign_loss"] = float(pixelalign_loss.item())

    if pointsdist_loss is not None:
        log_dict["pointsdist_loss"] = float(pointsdist_loss.item())

    if grad_norm is not None:
        log_dict["grad_norm"] = float(grad_norm)

    if optimizer is not None:
        log_dict["lr"] = float(optimizer.param_groups[0]["lr"])

    # [TODO] Add gaussians usage logging for 4DLRM, 
    # this is helpful for training-time monitoring and debugging.
    if gaussians_usage is not None:
        log_dict["gaussians_usage"] = float(gaussians_usage)

    wandb.log(log_dict, step=now_iters)

    
def log_pose_metrics(
    now_iters,
    epoch,
    rot_loss,
    transl_loss,
    intri_loss,
    rerr_deg,
    terr_deg,
    Racc_5,
    Racc_15,
    Racc_30,
    Tacc_5,
    Tacc_15,
    Tacc_30,
    mAA_30,
):
    """Log pose evaluation metrics to Weights & Biases.

    Args:
        now_iters: Current global iteration used as the logging step.
        epoch: Current epoch number.
        rot_loss: Rotation loss tensor.
        transl_loss: Translation loss tensor.
        intri_loss: Intrinsics loss tensor.
        rerr_deg: Per-sample rotation errors in degrees.
        terr_deg: Per-sample translation errors in degrees.
        Racc_5: Rotation accuracy at 5 degrees.
        Racc_15: Rotation accuracy at 15 degrees.
        Racc_30: Rotation accuracy at 30 degrees.
        Tacc_5: Translation accuracy at 5 degrees.
        Tacc_15: Translation accuracy at 15 degrees.
        Tacc_30: Translation accuracy at 30 degrees.
        mAA_30: Mean average accuracy up to 30 degrees.
    """
    log_dict = {
        "rot_loss": float(rot_loss.item()),
        "transl_loss": float(transl_loss.item()),
        "intri_loss": float(intri_loss.item()),
        "pose/Rerr_mean_deg": float(rerr_deg.mean().item()),
        "pose/Terr_mean_deg": float(terr_deg.mean().item()),
        "pose/Racc@5": float(Racc_5),
        "pose/Racc@15": float(Racc_15),
        "pose/Racc@30": float(Racc_30),
        "pose/Tacc@5": float(Tacc_5),
        "pose/Tacc@15": float(Tacc_15),
        "pose/Tacc@30": float(Tacc_30),
        "pose/mAA@30": float(mAA_30),
        "epoch": int(epoch),
    }
    wandb.log(log_dict, step=now_iters)


def _to_uint8_img(t: torch.Tensor) -> Image.Image:
    """Convert a CHW tensor to a uint8 PIL image.

    Floating-point tensors are clamped to [0, 1], scaled to [0, 255], and
    converted to uint8. Non-uint8 integer tensors are cast directly. Single-
    channel tensors are expanded to 3 channels, and tensors with more than
    3 channels are truncated to the first 3.

    Args:
        t: Image tensor of shape (C, H, W).

    Returns:
        Image.Image: PIL image in uint8 format.
    """
    t = t.detach().cpu()

    if t.dtype.is_floating_point:
        t = (t.clamp(0, 1) * 255.0).round().to(torch.uint8)
    elif t.dtype != torch.uint8:
        t = t.to(torch.uint8)

    c, h, w = t.shape
    if c == 1:
        t = t.expand(3, h, w)
    elif c != 3:
        t = t[:3]

    return Image.fromarray(t.permute(1, 2, 0).numpy())


def _per_sample_psnr(
    pred: torch.Tensor,
    gt: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute PSNR independently for each sample in a batch.

    Args:
        pred: Predicted images of shape (N, ...).
        gt: Ground-truth images of shape (N, ...).
        eps: Small constant added to the MSE for numerical stability.

    Returns:
        torch.Tensor: PSNR values of shape (N,).
    """
    mse = F.mse_loss(pred, gt, reduction="none").flatten(1).mean(dim=1)
    return -10.0 * torch.log10(mse + eps)


def _per_sample_ssim(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Compute per-sample SSIM between predicted and ground-truth images.

    Args:
        pred: (N, C, H, W) predicted images, value range [0, 1]
        gt:   (N, C, H, W) ground-truth images, value range [0, 1]

    Returns:
        Tensor of shape (N,), SSIM for each sample
    """
    pred_np = pred.detach().cpu().numpy()
    gt_np = gt.detach().cpu().numpy()

    ssim_vals = [
        ssim_fn(
            gt_i, pred_i,
            win_size=11,
            gaussian_weights=True,
            channel_axis=0,
            data_range=1.0
        )
        for gt_i, pred_i in zip(gt_np, pred_np)
    ]

    return torch.tensor(ssim_vals, dtype=pred.dtype, device=pred.device)


def _flatten_to_nchw(x: torch.Tensor) -> torch.Tensor:
    """Flatten a batch of images to NCHW format.

    Args:
        x: Image tensor of shape (B, V, C, H, W) or (N, C, H, W).

    Returns:
        torch.Tensor: Tensor of shape (N, C, H, W).

    Raises:
        ValueError: If the input shape is not supported.
    """
    if x.dim() == 5:
        b, v, c, h, w = x.shape
        return x.reshape(b * v, c, h, w)

    if x.dim() == 4:
        return x

    raise ValueError(f"Unexpected tensor shape {tuple(x.shape)}")


def _measure_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
):
    """Measure text size robustly across Pillow versions.

    Args:
        draw: Pillow drawing context.
        text: Text to measure.
        font: Font used for measurement.

    Returns:
        tuple: Text width and height as (w, h).
    """
    try:
        # Pillow >= 10
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        try:
            # Older Pillow versions.
            bbox = font.getbbox(text)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            # Last-resort fallback for legacy Pillow versions.
            return font.getsize(text)
