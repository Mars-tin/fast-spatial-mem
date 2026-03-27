
#
#  Apache 2.0 License License
#  Copyright (c) 2026 Martin Ziqiao Ma
#
import torch
import torch.nn.functional as F


def camera_se3_loss(
    pred_c2w,
    pred_fxfycxcy,
    gt_c2w,
    gt_fxfycxcy,
    W: int,
    H: int,
    normalize_gt: bool = True,
    focal_divisor: str = "maxwh",
    center_shift: bool = True,
    t_mode: str = "absolute",
    w_R: float = 1.0,
    w_t: float = 1.0,
    w_K: float = 1.0,
):
    """Compute pose and intrinsics losses for camera prediction.

    Args:
        pred_c2w: Predicted camera-to-world matrices of shape (B, 4, 4).
        pred_fxfycxcy: Predicted intrinsics of shape (B, 4) in the model's
            normalized output space.
        gt_c2w: Ground-truth camera-to-world matrices of shape (B, 3, 4) or
            (B, 4, 4).
        gt_fxfycxcy: Ground-truth intrinsics of shape (B, 4), typically in
            pixel coordinates.
        W: Image width. Used only when ``normalize_gt`` is True.
        H: Image height. Used only when ``normalize_gt`` is True.
        normalize_gt: Whether to normalize ground-truth intrinsics into the
            same space as the predictions.
        focal_divisor: Divisor mode passed to ``normalize_intrinsics_gt``.
        center_shift: Whether to apply center shifting in
            ``normalize_intrinsics_gt``.
        t_mode: Translation loss mode. Must be either ``"absolute"`` or
            ``"scale_invariant"``.
        w_R: Weight for the rotation loss.
        w_t: Weight for the translation loss.
        w_K: Weight for the intrinsics loss.

    Returns:
        dict: Dictionary containing total, rotation, translation, and
        intrinsics losses.
    """
    # Parse camera poses.
    R_pred, t_pred = parse_c2w_34_or_44(pred_c2w)
    R_gt, t_gt = parse_c2w_34_or_44(gt_c2w)

    # Use Smooth L1 on rotation matrices for now.
    l_R = F.smooth_l1_loss(R_pred, R_gt)

    # Translation loss.
    if t_mode == "absolute":
        l_t = F.smooth_l1_loss(t_pred, t_gt)
    elif t_mode == "scale_invariant":
        t_pred_u = F.normalize(t_pred, dim=-1)
        t_gt_u = F.normalize(t_gt, dim=-1)
        l_t = (1.0 - (t_pred_u * t_gt_u).sum(dim=-1)).mean()
    else:
        raise ValueError("t_mode should be in {'absolute','scale_invariant'}")

    # Intrinsics loss.
    if normalize_gt:
        gt_intr_n = normalize_intrinsics_gt(
            gt_fxfycxcy,
            W,
            H,
            focal_divisor=focal_divisor,
            center_shift=center_shift,
        )
    else:
        gt_intr_n = gt_fxfycxcy

    # Compare intrinsics directly in the model prediction space.
    l_K = F.smooth_l1_loss(pred_fxfycxcy, gt_intr_n)

    total = w_R * l_R + w_t * l_t + w_K * l_K

    # If any loss becomes NaN, inspect whether predictions contain NaNs.
    return {
        "total": total,
        "rot": l_R,
        "trans": l_t,
        "intr": l_K,
    }


def parse_c2w_34_or_44(c2w):
    """Parse camera-to-world matrices into rotation and translation.

    Args:
        c2w: Camera-to-world matrices of shape (..., 3, 4) or (..., 4, 4).

    Returns:
        tuple: A tuple of:
            - R: Rotation matrices of shape (..., 3, 3).
            - t: Translation vectors of shape (..., 3).

    Raises:
        ValueError: If the input shape is not (..., 3, 4) or (..., 4, 4).
    """
    if c2w.shape[-2:] == (4, 4):
        R = c2w[..., :3, :3]
        t = c2w[..., :3, 3]
    elif c2w.shape[-2:] == (3, 4):
        R = c2w[..., :3, :3]
        t = c2w[..., :3, 3]
    else:
        raise ValueError(f"Unsupported c2w shape: {c2w.shape}")

    return R, t


def normalize_intrinsics_gt(
    gt_fxfycxcy_pixels,
    W: int,
    H: int,
    focal_divisor: str = "maxwh",
    center_shift: bool = True,
):
    """Normalize ground-truth camera intrinsics.

    Args:
        gt_fxfycxcy_pixels: Ground-truth intrinsics of shape (B, 4) in pixel
            coordinates, ordered as [fx, fy, cx, cy].
        W: Image width.
        H: Image height.
        focal_divisor: Normalization divisor for focal lengths. Must be one of
            {"W", "H", "maxwh", "diag"}.
        center_shift: Whether to shift normalized principal points so that
            (0.5, 0.5) maps to (0.0, 0.0).

    Returns:
        torch.Tensor: Normalized intrinsics of shape (B, 4).
    """
    fx, fy, cx, cy = gt_fxfycxcy_pixels.unbind(-1)

    if focal_divisor == "W":
        scale = torch.as_tensor(W, dtype=fx.dtype, device=fx.device)
    elif focal_divisor == "H":
        scale = torch.as_tensor(H, dtype=fx.dtype, device=fx.device)
    elif focal_divisor == "diag":
        scale = torch.as_tensor(
            (W**2 + H**2) ** 0.5,
            dtype=fx.dtype,
            device=fx.device,
        )
    elif focal_divisor == "maxwh":
        scale = torch.as_tensor(
            max(W, H),
            dtype=fx.dtype,
            device=fx.device,
        )
    else:
        raise ValueError("focal_divisor must be one of {'W','H','maxwh','diag'}")

    fx_n = fx / scale
    fy_n = fy / scale
    cx_n = cx / W
    cy_n = cy / H

    if center_shift:
        cx_n = cx_n - 0.5
        cy_n = cy_n - 0.5

    return torch.stack([fx_n, fy_n, cx_n, cy_n], dim=-1)
