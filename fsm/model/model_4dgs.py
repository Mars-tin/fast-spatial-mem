# This file is derived from Gaussian-Splatting and is distributed under
# the terms of Gaussian-Splatting License.
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# For inquiries contact george.drettakis@inria.fr
#
# This file also incorporates material derived from code released by
# Fudan Zhang Vision Group under the MIT License.
#
# Copyright (c) 2024 Fudan Zhang Vision Group
#
# This file also incorporates material derived from code released by
# Ziwen Chen under the Apache 2.0 License.
#
# Copyright (c) 2025 Ziwen Chen
#
# Modifications Copyright (c) 2026, Martin Ziqiao Ma
#
import math
from typing import Optional
import copy

import torch
from torch import nn

# from plyfile import PlyData, PlyElement
from types import SimpleNamespace
from collections import OrderedDict

from .diff_gaussian_rasterization import GaussianRasterizer
from .utils_sh import SH2RGB

# =============================================================================
# 4D Gaussian Splatting Utilities
# =============================================================================


def strip_symmetric(L):
    """Extract the upper-triangular entries of a symmetric 3x3 matrix.

    Args:
        L: Tensor of shape [N, 3, 3].

    Returns:
        torch.Tensor: Tensor of shape [N, 6] containing
        [L00, L01, L02, L11, L12, L22].
    """
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device=L.device)
    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def build_rotation(r):
    """Convert quaternions to rotation matrices.

    Args:
        r: Quaternion tensor of shape [N, 4] in (w, x, y, z) order.

    Returns:
        torch.Tensor: Rotation matrices of shape [N, 3, 3].
    """
    norm = torch.sqrt(
        r[:, 0] * r[:, 0]
        + r[:, 1] * r[:, 1]
        + r[:, 2] * r[:, 2]
        + r[:, 3] * r[:, 3]
    )
    q = r / norm[:, None]
    R = torch.zeros((q.size(0), 3, 3), device=r.device)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)

    return R


def build_scaling_rotation(s, r):
    """Build a scaled rotation matrix.

    Args:
        s: Scaling factors of shape [N, 3].
        r: Quaternion tensor of shape [N, 4].

    Returns:
        torch.Tensor: Scaled rotation matrices of shape [N, 3, 3].
    """
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device=s.device)
    R = build_rotation(r)
    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]
    L = R @ L
    return L


def build_rotation_4d(l, r):
    """Build 4D rotation matrices from pairs of quaternions.

    Args:
        l: Left quaternion tensor of shape [N, 4].
        r: Right quaternion tensor of shape [N, 4].

    Returns:
        torch.Tensor: 4D rotation matrices of shape [N, 4, 4].
    """
    l_norm = torch.norm(l, dim=-1, keepdim=True)
    r_norm = torch.norm(r, dim=-1, keepdim=True)

    q_l = l / l_norm
    q_r = r / r_norm
    a, b, c, d = q_l.unbind(-1)
    p, q, r, s = q_r.unbind(-1)

    M_l = torch.stack([a,-b,-c,-d,
                       b, a,-d, c,
                       c, d, a,-b,
                       d,-c, b, a]).view(4,4,-1).permute(2,0,1)
    
    M_r = torch.stack([ p, q, r, s,
                       -q, p,-s, r,
                       -r, s, p,-q,
                       -s,-r, q, p]).view(4,4,-1).permute(2,0,1)
    
    A = M_l @ M_r
    A = A.flip(1,2)
    return A


def build_scaling_rotation_4d(s, l, r):
    """Build scaled 4D rotation matrices.

    Args:
        s: Scaling factors of shape [N, 4].
        l: Left quaternion tensor of shape [N, 4].
        r: Right quaternion tensor of shape [N, 4].

    Returns:
        torch.Tensor: Scaled 4D rotation matrices of shape [N, 4, 4].
    """
    L = torch.zeros((s.shape[0], 4, 4), dtype=torch.float, device=s.device)
    R = build_rotation_4d(l, r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]
    L[:,3,3] = s[:,3]

    L = R @ L
   
    return L


# =============================================================================
# Camera Wrapper
# =============================================================================


class Camera(nn.Module):
    """Camera wrapper that stores extrinsics, intrinsics, and projection matrices."""

    def __init__(self, c2w, fxfycxcy, h, w, znear=0.01, zfar=100.0):
        """Initialize the camera.

        Args:
            c2w: Camera-to-world matrix of shape (4, 4) in OpenCV convention.
            fxfycxcy: Camera intrinsics [fx, fy, cx, cy].
            h: Image height.
            w: Image width.
            znear: Near clipping plane.
            zfar: Far clipping plane.
        """
        super().__init__()

        self.c2w = c2w.clone().float()
        self.w2c = self.c2w.inverse()
        self.h = h
        self.w = w
        self.znear = znear
        self.zfar = zfar

        fx, fy, cx, cy = fxfycxcy[0], fxfycxcy[1], fxfycxcy[2], fxfycxcy[3]

        self.tanfovX = w / (2 * fx)
        self.tanfovY = h / (2 * fy)

        # Build the perspective projection matrix from pinhole intrinsics
        projection_matrix = torch.zeros(4, 4, device=fx.device)
        projection_matrix[0, 0] = 2 * fx / w
        projection_matrix[1, 1] = 2 * fy / h
        projection_matrix[0, 2] = 2 * (cx / w) - 1
        projection_matrix[1, 2] = 2 * (cy / h) - 1
        projection_matrix[2, 2] = -(zfar + znear) / (zfar - znear)
        projection_matrix[3, 2] = 1.0
        projection_matrix[2, 3] = -(2 * zfar * znear) / (zfar - znear)

        self.world_view_transform = self.w2c.transpose(0, 1)
        self.projection_matrix = projection_matrix.transpose(0, 1)
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0)
            .bmm(self.projection_matrix.unsqueeze(0))
            .squeeze(0)
        )
        self.camera_center = self.c2w[:3, 3]


# =============================================================================
# 4D Gaussian Splatting Model
# =============================================================================


class GaussianModel:
    """
    4DGS Renderer
    Modified from https://github.com/fudan-zvg/4d-gaussian-splatting/blob/main/scene/gaussian_model.py
    """

    def setup_functions(self):
        
        # For 3DGS
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        # For 4DGS
        def build_covariance_from_scaling_rotation_4d(scaling, scaling_modifier, rotation_l, rotation_r, dt=0.0):
            L = build_scaling_rotation_4d(scaling_modifier * scaling, rotation_l, rotation_r)
            actual_covariance = L @ L.transpose(1, 2)
            cov_11 = actual_covariance[:,:3,:3]
            cov_12 = actual_covariance[:,0:3,3:4]
            cov_t = actual_covariance[:,3:4,3:4]
            current_covariance = cov_11 - cov_12 @ cov_12.transpose(1, 2) / cov_t
            symm = strip_symmetric(current_covariance)
            if dt.shape[1] > 1:
                mean_offset = (cov_12.squeeze(-1) / cov_t.squeeze(-1))[:, None, :] * dt[..., None]
                mean_offset = mean_offset[..., None]  # [num_pts, num_time, 3, 1]
            else:
                mean_offset = cov_12.squeeze(-1) / cov_t.squeeze(-1) * dt
            return symm, mean_offset.squeeze(-1)
        
        self.scaling_activation = torch.exp
        self.inv_scaling_activation = torch.log
        self.rotation_activation = torch.nn.functional.normalize
        self.opacity_activation = torch.sigmoid

        if self.rot_4d: self.covariance_activation = build_covariance_from_scaling_rotation_4d
        else: self.covariance_activation = build_covariance_from_scaling_rotation

    def __init__(
        self, 
        gaussian_dim : int = 4, 
        scaling_modifier: Optional[float] = None,
        sh_degree : int = 0, 
        sh_degree_t : int = 0, 
        force_sh_3d: bool = False,
        time_duration: list = [-1.0, 1.0], 
        rot_4d: bool = True
    ):

        self.sh_degree = sh_degree
        self.sh_degree_t = sh_degree_t

        self._xyz = torch.empty(0)
        self._t = torch.empty(0)

        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0) if self.sh_degree > 0 else None

        self._scaling = torch.empty(0)
        self._scaling_t = torch.empty(0)
        self._rotation = torch.empty(0)
        self._rotation_r = torch.empty(0)
        self._opacity = torch.empty(0)

        self.gaussian_dim = gaussian_dim
        self.time_duration = time_duration
        self.rot_4d = rot_4d
        self.force_sh_3d = force_sh_3d
        if self.rot_4d or self.force_sh_3d:
            assert self.gaussian_dim == 4

        self.setup_functions()

        self.scaling_modifier = scaling_modifier

    def reset(self):
        self.__init__(
            self.gaussian_dim,
            self.scaling_modifier,
            self.sh_degree,
            self.sh_degree_t,
            self.force_sh_3d,
            self.time_duration,
            self.rot_4d
        )

    def set_gs_params(self, xyz, features, scaling, rotation, opacity, t=None, scaling_t=None, rotation_r=None):
        """
        xyz : torch.tensor of shape (N, 3)
        features : torch.tensor of shape (N, (self.sh_degree + 1) ** 2, 3)
        scaling : torch.tensor of shape (N, 3)
        rotation : torch.tensor of shape (N, 4)
        opacity : torch.tensor of shape (N, 1)
        t : torch.tensor of shape (N, 1)
        """
        self._xyz = xyz
        self._features_dc = features[:, :1, :].contiguous()
        if self.sh_degree > 0:
            self._features_rest = features[:, 1:, :].contiguous()
        else:
            self._features_rest = None
        self._scaling = scaling
        self._rotation = rotation
        self._opacity = opacity
        if t is not None:
            self._t = t
            self._scaling_t = scaling_t
            self._rotation_r = rotation_r
        return self

    def to(self, device):
        self._xyz = self._xyz.to(device)
        self._features_dc = self._features_dc.to(device)
        if self.sh_degree > 0:
            self._features_rest = self._features_rest.to(device)
        self._scaling = self._scaling.to(device)
        self._rotation = self._rotation.to(device)
        self._opacity = self._opacity.to(device)
        self._t = self._t.to(device)
        self._scaling_t = self._scaling_t.to(device)
        self._rotation_r = self._rotation_r.to(device)
        return self

    def filter(self, valid_mask):
        self._xyz = self._xyz[valid_mask]
        self._features_dc = self._features_dc[valid_mask]
        if self.sh_degree > 0:
            self._features_rest = self._features_rest[valid_mask]
        self._scaling = self._scaling[valid_mask]
        self._rotation = self._rotation[valid_mask]
        self._opacity = self._opacity[valid_mask]

        if self.gaussian_dim == 4:
            self._t = self._t[valid_mask]
            self._scaling_t = self._scaling_t[valid_mask]
            if self.rot_4d:
                self._rotation_r = self._rotation_r[valid_mask]
        return self

    def prune_by_bbox(self, crop_bbx=[-1, 1, -1, 1, -1, 1]):
        x_min, x_max, y_min, y_max, z_min, z_max = crop_bbx
        xyz = self._xyz
        invalid_mask = (
            (xyz[:, 0] < x_min)
            | (xyz[:, 0] > x_max)
            | (xyz[:, 1] < y_min)
            | (xyz[:, 1] > y_max)
            | (xyz[:, 2] < z_min)
            | (xyz[:, 2] > z_max)
        )
        valid_mask = ~invalid_mask
        return self.filter(valid_mask)

    def prune_by_opacity(self, opacity_thres=0.05):
        opacity = self.get_opacity.squeeze(1)
        valid_mask = opacity > opacity_thres
        return self.filter(valid_mask)

    def prune_by_rgb(self, rgb_white_thres=20.0):
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        rgb = (SH2RGB(f_dc) * 255.0).clip(0.0, 255.0)
        rgb_mask = (rgb < 255-rgb_white_thres).any(axis=1)  # True if not pure white
        return self.filter(rgb_mask)

    def prune_by_timespan(self, timestamp, pt_thres=0.05):
        marginal_t = self.get_marginal_t(timestamp)
        valid_mask = marginal_t[:,0] > pt_thres
        return self.filter(valid_mask)
    
    def prune_by_nearfar(self, cam_origins, nearfar_percent=(0.01, 0.99)):
        assert len(nearfar_percent) == 2
        assert nearfar_percent[0] < nearfar_percent[1]
        assert nearfar_percent[0] >= 0 and nearfar_percent[1] <= 1
        device = self._xyz.device
        dists = torch.cdist(self._xyz[None], cam_origins[None].to(device))[0]
        dists_percentile = torch.quantile(
            dists, torch.tensor(nearfar_percent).to(device), dim=0
        )
        reject_mask = (dists < dists_percentile[0:1, :]) | (
            dists > dists_percentile[1:2, :]
        )
        reject_mask = reject_mask.any(dim=1)
        valid_mask = ~reject_mask
        return self.filter(valid_mask)

    # For 3DGS
    @property
    def get_scaling(self):
        if self.scaling_modifier is not None:
            return self.scaling_activation(self._scaling) * self.scaling_modifier
        else:
            return self.scaling_activation(self._scaling)
        
    # For 4DGS
    @property
    def get_scaling_t(self):
        if self.scaling_modifier is not None:
            return self.scaling_activation(self._scaling_t) * self.scaling_modifier
        else:
            return self.scaling_activation(self._scaling_t)
        
    # For 4DGS 
    @property
    def get_scaling_xyzt(self):
        if self.scaling_modifier is not None:
            return self.scaling_activation(torch.cat([self._scaling, self._scaling_t], dim = 1)) * self.scaling_modifier
        else:
            return self.scaling_activation(torch.cat([self._scaling, self._scaling_t], dim = 1))
    
    # For 3D/4DGS
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    # For 4DGS 
    @property
    def get_rotation_r(self):
        return self.rotation_activation(self._rotation_r)
    
    # For 3D/4DGS
    @property
    def get_xyz(self):
        return self._xyz
    
    # For 4DGS 
    @property
    def get_t(self):
        return self._t
    
    # For 4DGS 
    @property
    def get_xyzt(self):
        return torch.cat([self._xyz, self._t], dim = 1)
    
    # For 3D/4DGS
    @property
    def get_features(self):
        if self.sh_degree > 0:
            features_dc = self._features_dc
            features_rest = self._features_rest
            return torch.cat((features_dc, features_rest), dim=1)
        else:
            return self._features_dc
        
    # For 3D/4DGS
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    # For 4DGS
    @property
    def get_max_sh_channels(self):
        if self.gaussian_dim == 3 or self.force_sh_3d:
            return (self.sh_degree+1)**2
        elif self.gaussian_dim == 4 and self.sh_degree_t == 0:
            sh_channels_4d = [1, 6, 16, 33]
            return sh_channels_4d[self.sh_degree]
        elif self.gaussian_dim == 4 and self.sh_degree_t > 0:
            return (self.sh_degree+1)**2 * (self.sh_degree_t + 1)
        
    # For 3DGS
    def get_covariance(self, scaling_modifier=1):
        assert self.gaussian_dim == 3 and not self.rot_4d
        return self.covariance_activation(
            self.get_scaling, 
            scaling_modifier, 
            self._rotation
        )
    
    # For 4DGS
    def get_current_covariance_and_mean_offset(self, scaling_modifier=1, timestamp=0.0):
        assert self.gaussian_dim == 4 and self.rot_4d
        return self.covariance_activation(
            self.get_scaling_xyzt, 
            scaling_modifier, 
            self._rotation, 
            self._rotation_r,
            dt = timestamp - self.get_t
        )
    
    # For 4DGS
    def get_cov_t(self, scaling_modifier=1):
        if self.rot_4d:
            L = build_scaling_rotation_4d(scaling_modifier * self.get_scaling_xyzt, self._rotation, self._rotation_r)
            actual_covariance = L @ L.transpose(1, 2)
            return actual_covariance[:,3,3].unsqueeze(1)
        else:
            return self.get_scaling_t * scaling_modifier
        
    # For 4DGS
    def get_marginal_t(self, timestamp, scaling_modifier = 1):

        # [NOTE] I kept the below comments from 4DGS Authors
        # 这里乘完会大于1，绝对不行.
        # marginal_t应该用个概率而非概率密度 暂时可以clamp一下，后期用积分
        # marginal_t = torch.clamp_max(marginal_t, 1.0)

        sigma = self.get_cov_t(scaling_modifier)
        marginal_t = torch.exp(-0.5*(self.get_t-timestamp)**2/sigma)
        return marginal_t

    def save_ply(self):
        # [TODO] Add PLY saving for 4DGS (with t, scaling_t, rotation_r)
        raise NotImplementedError

    def load_ply(self):
        # [TODO] Add PLY loading for 4DGS
        raise NotImplementedError


# =============================================================================
# Deferred Gaussian Splatting Renderer with Autograd Support
# =============================================================================


class DeferredGaussianRender(torch.autograd.Function):
    """
    4DGS DeferredRender
    Modified from https://github.com/arthurhero/Long-LRM/blob/main/model/llrm.py
    """

    @staticmethod
    def forward(
        ctx,
        xyz,
        t,
        features,
        scaling,
        scaling_t,
        rotation,
        rotation_r,
        opacity,
        height,
        width,
        c2w,
        fxfycxcy,
        timestamp,
        scaling_modifier=None,
        t_prune=False,
        t_threshold=0.05,
    ):
        """Render deferred Gaussian splats.

        Args:
            ctx: Autograd context.
            xyz: Gaussian centers of shape [B, N, 3].
            t: Gaussian timestamps of shape [B, N, 1] or compatible shape.
            features: Gaussian features of shape [B, N, (sh_degree + 1)^2, 3].
            scaling: Spatial scales of shape [B, N, 3].
            scaling_t: Temporal scales of shape [B, N, 1] or compatible shape.
            rotation: Spatial rotations of shape [B, N, 4].
            rotation_r: Additional 4D rotation parameters of shape [B, N, 4].
            opacity: Gaussian opacities of shape [B, N, 1].
            height: Output image height.
            width: Output image width.
            c2w: Camera-to-world matrices of shape [B, V, 4, 4].
            fxfycxcy: Camera intrinsics of shape [B, V, 4].
            timestamp: Target timestamps of shape [B, V, 1].
            scaling_modifier: Optional scaling modifier passed to the Gaussian model.
            t_prune: Whether to temporally prune Gaussians before rendering.
            t_threshold: Temporal pruning threshold.

        Returns:
            torch.Tensor: Rendered images of shape [B, V, 3, height, width].
        """

        ctx.scaling_modifier = scaling_modifier
        sh_degree = int(math.sqrt(features.shape[-2])) - 1
        gaussians_model = GaussianModel(
            gaussian_dim=4,
            scaling_modifier=scaling_modifier,
            sh_degree=sh_degree, 
        )

        with torch.no_grad():
            batch_size, num_views = c2w.size(0), c2w.size(1)
            renderings = []
            for i in range(batch_size):
                pc = gaussians_model.set_gs_params(
                    xyz[i], features[i], scaling[i], rotation[i], opacity[i], t[i], scaling_t[i], rotation_r[i]
                )
                for j in range(num_views):
                    if t_prune:
                        pc_pruned = t_slice(pc, timestamp[i, j], t_threshold)
                        renderings.append(
                            _rasterizer(pc_pruned, height, width, c2w[i, j], fxfycxcy[i, j], timestamp[i, j])["rendering"]
                        )
                    else:
                        renderings.append(
                            _rasterizer(pc, height, width, c2w[i, j], fxfycxcy[i, j], timestamp[i, j])["rendering"]
                        )
            renderings = torch.stack(renderings, dim=0)
            renderings = renderings.reshape(batch_size, num_views, 3, height, width)

        renderings = renderings.requires_grad_()
        ctx.save_for_backward(
            xyz, t, features, scaling, scaling_t, rotation, rotation_r, opacity, c2w, fxfycxcy, timestamp
        )
        ctx.rendering_size = (height, width)
        ctx.sh_degree = sh_degree

        del gaussians_model

        return renderings

    @staticmethod
    def backward(ctx, grad_output):
        """Backpropagate through deferred Gaussian rendering.

        Args:
            ctx: Autograd context.
            grad_output: Gradient of the rendered output of shape
                [B, V, 3, height, width].

        Returns:
            Tuple of gradients matching the forward argument order.
        """
        xyz, t, features, scaling, scaling_t, rotation, rotation_r, opacity, c2w, fxfycxcy, timestamp = ctx.saved_tensors
        height, width = ctx.rendering_size
        sh_degree = ctx.sh_degree

        gs_params = OrderedDict(
            xyz=xyz,
            t=t,
            features=features,
            scaling=scaling,
            scaling_t=scaling_t,
            rotation=rotation,
            rotation_r=rotation_r,
            opacity=opacity,
        )
        gs_params = OrderedDict(
            (key, value.detach().requires_grad_())
            for key, value in gs_params.items()
        )
        gaussians_model = GaussianModel(
            gaussian_dim=4, 
            scaling_modifier=ctx.scaling_modifier, 
            sh_degree=sh_degree
        )

        with torch.enable_grad():
            b, v = c2w.size(0), c2w.size(1)
            for i in range(b):
                for j in range(v):
                    pc = gaussians_model.set_gs_params(**{k: v[i] for k, v in gs_params.items()})
                    rendering = _rasterizer(pc, height, width, c2w[i, j], fxfycxcy[i, j], timestamp[i, j])["rendering"]
                    rendering.backward(grad_output[i, j])

        del rendering
        # torch.cuda.empty_cache()

        return *[var.grad for var in gs_params.values()], None, None, None, None, None, None, None, None


# =============================================================================
# Utility Functions for Rendering and Pruning
# =============================================================================


@torch.no_grad()
def t_slice(pc: GaussianModel, timestamp, t_threshold=0.05):
    sliced_pc = copy.deepcopy(pc).prune_by_timespan(timestamp, t_threshold)
    return sliced_pc


def _rasterizer(
    pc: GaussianModel,
    height: int,
    width: int,
    c2w: torch.Tensor,
    fxfycxcy: torch.Tensor,
    timestamp: torch.Tensor,
    bg_color=(1.0, 1.0, 1.0),
    scaling_modifier=1.0,
    znear=0.01,
    zfar=100.0
):
    """
    Render the scene with opencv cameras.
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor for the gradients of the 2D (screen-space) means
    viewspace_points = torch.empty_like(
        pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
    )
    camera = Camera(c2w=c2w, fxfycxcy=fxfycxcy, h=height, w=width, znear=znear, zfar=zfar)
    bg_color = torch.tensor(list(bg_color), dtype=torch.float32, device=c2w.device)

    # Set up rasterization configuration
    rasterizer = GaussianRasterizer(
        raster_settings=SimpleNamespace(
            image_height=int(camera.h),
            image_width=int(camera.w),
            tanfovx=camera.tanfovX.item(),
            tanfovy=camera.tanfovY.item(),
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=camera.world_view_transform,
            projmatrix=camera.full_proj_transform,
            sh_degree=pc.sh_degree,
            sh_degree_t=pc.sh_degree_t,
            campos=camera.camera_center,
            timestamp=timestamp.item(),
            time_duration=pc.time_duration[1]-pc.time_duration[0],
            rot_4d=pc.rot_4d,
            gaussian_dim=pc.gaussian_dim,
            force_sh_3d=pc.force_sh_3d,
            prefiltered=False,
            debug=False
        )
    )

    means3D = pc.get_xyz
    means2D = viewspace_points
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    shs = pc.get_features
    flow_2d = torch.zeros_like(means3D[:, :2])
    ts = None
    scales_t = None
    rotations_r = None

    if pc.gaussian_dim == 4:
        ts = pc.get_t
        scales_t = pc.get_scaling_t
        if pc.rot_4d:
            rotations_r = pc.get_rotation_r

    rendering, radii, depth, alpha, flow, covs_com = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=None,
        flow_2d=flow_2d,
        opacities=opacity,
        ts=ts,
        scales=scales,
        scales_t=scales_t,
        rotations=rotations,
        rotations_r=rotations_r,
        cov3D_precomp=None,
    )

    return {
        "rendering": rendering,
        # "viewspace_points": viewspace_points,
        # "radii": radii,
        # "depth": depth,
        # "alpha": alpha,
        # "flow": flow,
        # "covs_com": covs_com
    }
