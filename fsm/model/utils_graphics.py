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
import torch
import torch.nn.functional as F


def compute_rays(fxfycxcy, c2w, h, w):
    """Compute rays for each pixel in the image.

    Args:
        fxfycxcy (torch.tensor): [b, v, 4]
        c2w (torch.tensor): [b, v, 4, 4]
        
    Returns:
        ray_o: (b, v, 3, h, w)
        ray_d: (b, v, 3, h, w)
    """
    b, v = fxfycxcy.size(0), fxfycxcy.size(1)

    # Efficient meshgrid equivalent using broadcasting
    idx_x = torch.arange(w, device=c2w.device)[None, :].expand(h, -1)  # [h, w]
    idx_y = torch.arange(h, device=c2w.device)[:, None].expand(-1, w)  # [h, w]

    # Reshape for batched matrix multiplication
    idx_x = idx_x.flatten().expand(b * v, -1)           # [b*v, h*w]
    idx_y = idx_y.flatten().expand(b * v, -1)           # [b*v, h*w]

    fxfycxcy = fxfycxcy.reshape(b * v, 4)               # [b*v, 4]
    c2w = c2w.reshape(b * v, 4, 4)                      # [b*v, 4, 4]

    x = (idx_x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]     # [b*v, h*w]
    y = (idx_y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]     # [b*v, h*w]
    z = torch.ones_like(x)                                      # [b*v, h*w]

    ray_d = torch.stack([x, y, z], dim=1)                       # [b*v, 3, h*w]
    ray_d = torch.bmm(c2w[:, :3, :3], ray_d)                    # [b*v, 3, h*w]
    ray_d = ray_d / torch.norm(ray_d, dim=1, keepdim=True)      # [b*v, 3, h*w]

    ray_o = c2w[:, :3, 3:4].expand(b * v, -1, h*w)              # [b*v, 3, h*w]

    ray_o = ray_o.reshape(b, v, 3, h, w)                        # [b, v, 3, h, w]
    ray_d = ray_d.reshape(b, v, 3, h, w)                        # [b, v, 3, h, w]

    return ray_o, ray_d


def rot6d2mat(x):
    """Convert 6D rotation representations to rotation matrices.

    Based on Zhou et al., "On the Continuity of Rotation Representations in
    Neural Networks", CVPR 2019.

    Args:
        x: Rotation representations of shape [B, 6].

    Returns:
        torch.Tensor: Rotation matrices of shape [B, 3, 3].
    """
    a1 = x[:, 0:3]
    a2 = x[:, 3:6]

    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(
        a2 - (b1 * a2).sum(-1, keepdim=True) * b1,
        dim=-1,
    )
    b3 = torch.cross(b1, b2, dim=1)

    return torch.stack((b1, b2, b3), dim=-1)


def quat2mat(quat):
    """Convert quaternions to rotation matrices.

    Args:
        quat: Quaternion coefficients of shape [B, 4].

    Returns:
        torch.Tensor: Rotation matrices of shape [B, 3, 3].
    """
    norm_quat = quat / quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = (
        norm_quat[:, 0],
        norm_quat[:, 1],
        norm_quat[:, 2],
        norm_quat[:, 3],
    )

    batch_size = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rot_mat = torch.stack(
        [
            w2 + x2 - y2 - z2,
            2 * xy - 2 * wz,
            2 * wy + 2 * xz,
            2 * wz + 2 * xy,
            w2 - x2 + y2 - z2,
            2 * yz - 2 * wx,
            2 * xz - 2 * wy,
            2 * wx + 2 * yz,
            w2 - x2 - y2 + z2,
        ],
        dim=1,
    ).view(batch_size, 3, 3)
    return rot_mat


def get_cam_se3(cam_info):
    """Convert camera pose parameters into a 4x4 camera-to-world matrix.

    Args:
        cam_info: Tensor of shape [B, N], containing rotation, translation,
            and intrinsics parameters. Supported layouts are:
            - N == 13: [rot_6d(6), trans(3), fxfycxcy(4)]
            - N == 11: [rot_quat(4), trans(3), fxfycxcy(4)]

    Returns:
        tuple: A tuple of:
            - c2w: Camera-to-world matrices of shape [B, 4, 4].
            - fxfycxcy: Intrinsics parameters of shape [B, 4].

    Raises:
        NotImplementedError: If the input feature dimension is unsupported.
    """
    batch_size, num_features = cam_info.shape

    if num_features == 13:
        rot_6d = cam_info[:, :6]
        R = rot6d2mat(rot_6d)
        t = cam_info[:, 6:9].unsqueeze(-1)
        fxfycxcy = cam_info[:, 9:]
    elif num_features == 11:
        rot_quat = cam_info[:, :4]
        R = quat2mat(rot_quat)
        t = cam_info[:, 4:7].unsqueeze(-1)
        fxfycxcy = cam_info[:, 7:]
    else:
        raise NotImplementedError

    Rt = torch.cat([R, t], dim=2)
    bottom_row = torch.tensor(
        [0, 0, 0, 1],
        dtype=R.dtype,
        device=R.device,
    ).view(1, 1, 4).repeat(batch_size, 1, 1)
    c2w = torch.cat([Rt, bottom_row], dim=1)

    return c2w, fxfycxcy


def _canonicalize_quat_wpos(q: torch.Tensor) -> torch.Tensor:
    """Canonicalize quaternions so the scalar component is non-negative.

    Args:
        q: Quaternions of shape (..., 4) in (w, x, y, z) order.

    Returns:
        torch.Tensor: Canonicalized quaternions with w >= 0.
    """
    # Flip the sign when w < 0 so q and -q share a consistent representation.
    sign = torch.where(q[..., :1] < 0, -1.0, 1.0)
    return q * sign


def _rotmat_to_quat_wxyz(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to canonicalized quaternions.

    Args:
        R: Rotation matrices of shape (..., 3, 3).

    Returns:
        torch.Tensor: Quaternions of shape (..., 4) in (w, x, y, z) order,
        normalized and canonicalized to have non-negative w.
    """
    t = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    w = torch.sqrt(torch.clamp(t + 1.0, min=1e-8)) * 0.5
    x = (R[..., 2, 1] - R[..., 1, 2]) / (4.0 * w + 1e-8)
    y = (R[..., 0, 2] - R[..., 2, 0]) / (4.0 * w + 1e-8)
    z = (R[..., 1, 0] - R[..., 0, 1]) / (4.0 * w + 1e-8)

    q = torch.stack([w, x, y, z], dim=-1)
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
    return _canonicalize_quat_wpos(q)


def _quat_geodesic_angle(q_pred: torch.Tensor, q_gt: torch.Tensor) -> torch.Tensor:
    """Compute the geodesic angle between quaternions.

    Args:
        q_pred: Predicted quaternions of shape (..., 4) in (w, x, y, z) order.
        q_gt: Ground-truth quaternions of shape (..., 4) in (w, x, y, z) order.

    Returns:
        torch.Tensor: Geodesic angles in radians with shape (...,).
    """
    # Inputs are assumed to be normalized and canonicalized. Use the absolute
    # dot product so the angle is invariant to the q / -q double cover.
    dot = torch.sum(q_pred * q_gt, dim=-1)
    dot = torch.clamp(dot, -1.0, 1.0).abs()
    return 2.0 * torch.arccos(dot)

def _huber(x: torch.Tensor, delta: float) -> torch.Tensor:
    """Compute the elementwise Huber penalty.

    Args:
        x: Input tensor.
        delta: Huber transition point. If non-positive, this reduces to L1.

    Returns:
        torch.Tensor: Elementwise Huber penalty with the same shape as ``x``.
    """
    if delta <= 0:
        return x.abs()

    abs_x = x.abs()
    return torch.where(
        abs_x < delta,
        0.5 * (abs_x**2) / delta,
        abs_x - 0.5 * delta,
    )


def _split_views(tensor: torch.Tensor, v_in: int, v_out: int):
    """Split a view-major tensor into input and output views.

    Args:
        tensor: Tensor of shape [B, V_total, ...].
        v_in: Number of input views to take from the front.
        v_out: Number of output views to take from the back.

    Returns:
        tuple: A tuple of:
            - Tensor of shape [B, v_in, ...] from the first views.
            - Tensor of shape [B, v_out, ...] from the last views.
    """
    return tensor[:, :v_in], tensor[:, -v_out:]


def _relative_from_first(R_abs: torch.Tensor, t_abs: torch.Tensor):
    """Convert absolute poses to poses relative to the first view.

    Args:
        R_abs: Absolute rotations of shape [B, V, 3, 3].
        t_abs: Absolute translations of shape [B, V, 3].

    Returns:
        tuple: A tuple of:
            - Relative rotations of shape [B, V, 3, 3].
            - Relative translations of shape [B, V, 3].
    """
    R0 = R_abs[:, :1]
    t0 = t_abs[:, :1]

    # Express each pose relative to the first view in the sequence.
    R_rel = torch.matmul(R_abs, R0.transpose(-1, -2))
    t_rel = t_abs - t0

    return R_rel, t_rel


def fxy_to_fov(fx: torch.Tensor, fy: torch.Tensor, W: int, H: int):
    """Convert focal lengths in pixels to horizontal and vertical field of view.

    Args:
        fx: Horizontal focal lengths.
        fy: Vertical focal lengths.
        W: Image width in pixels.
        H: Image height in pixels.

    Returns:
        tuple: A tuple of:
            - Horizontal field of view in radians.
            - Vertical field of view in radians.
    """
    theta_x = 2.0 * torch.atan(W / (2.0 * fx.clamp_min(1e-6)))
    theta_y = 2.0 * torch.atan(H / (2.0 * fy.clamp_min(1e-6)))
    return theta_x, theta_y


def s_to_fov(s: torch.Tensor):
    """Convert log-tangent parameterization to field of view.

    Args:
        s: Log-tangent parameterization, where s = log(tan(theta / 2)).

    Returns:
        torch.Tensor: Field of view in radians.
    """
    return 2.0 * torch.atan(torch.exp(s))


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternions to rotation matrices.

    Args:
        q: Quaternions of shape (..., 4) in (w, x, y, z) order.

    Returns:
        torch.Tensor: Rotation matrices of shape (..., 3, 3).
    """
    w, x, y, z = q.unbind(-1)
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    rotation = torch.stack(
        [
            ww + xx - yy - zz,
            2 * (xy - wz),
            2 * (xz + wy),
            2 * (xy + wz),
            ww - xx + yy - zz,
            2 * (yz - wx),
            2 * (xz - wy),
            2 * (yz + wx),
            ww - xx - yy + zz,
        ],
        dim=-1,
    )
    return rotation.reshape(q.shape[:-1] + (3, 3))


def pose7_to_Rt(pose7: torch.Tensor):
    """Convert 7D pose parameters to rotation and translation.

    Args:
        pose7: Pose tensor of shape (..., 7), formatted as [T(3), q(4)] where
            the quaternion is in (w, x, y, z) order.

    Returns:
        tuple: A tuple of:
            - Rotation matrices of shape (..., 3, 3).
            - Translation vectors of shape (..., 3).
    """
    translation = pose7[..., :3]
    quat = pose7[..., 3:7]
    quat = quat / (quat.norm(dim=-1, keepdim=True) + 1e-8)

    # Canonicalize quaternion sign so q and -q map consistently.
    sign = torch.where(quat[..., :1] < 0, -1.0, 1.0)
    quat = quat * sign

    rotation = quat_to_rotmat(quat)
    return rotation, translation
