#
# This file is derived from code released under the MIT License.
#
# Original copyright:
# Copyright (c) 2024 Fudan Zhang Vision Group
#
# Modifications copyright:
# Copyright (c) 2026 Martin Ziqiao Ma
#
# This file is distributed under the licensing terms provided in the
# repository LICENSE and NOTICE files.
#
import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

parent_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
    "model/diff-gaussian-rasterization"
)

# [TODO]: Package this CUDA extension as a prebuilt module instead of compiling it
# at import time with `torch.utils.cpp_extension.load()`.
#
# Current behavior:
# - The extension is JIT-compiled when this module is imported.
# - This requires a CUDA-capable system and a compatible local CUDA toolkit.
# - The debug flags below are kept intentionally for CUDA-side debugging.
_C = load(
    name="diff_gaussian_rasterization",
    extra_cuda_cflags=[
        "-I " + os.path.join(parent_dir, "third_party/glm/"),
        "-g",
        "-DTORCH_USE_CUDA_DSA",  # Enable CUDA device-side assertions for debugging.
    ],
    sources=[
        os.path.join(parent_dir, "cuda_rasterizer/rasterizer_impl.cu"),
        os.path.join(parent_dir, "cuda_rasterizer/forward.cu"),
        os.path.join(parent_dir, "cuda_rasterizer/backward.cu"),
        os.path.join(parent_dir, "rasterize_points.cu"),
        os.path.join(parent_dir, "ext.cpp"),
    ],
    verbose=True,
)


class _RasterizeGaussians(torch.autograd.Function):
    """Autograd wrapper for the CUDA Gaussian rasterizer."""

    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        flow_2d,
        opacities,
        ts,
        scales,
        scales_t,
        rotations,
        rotations_r,
        cov3Ds_precomp,
        raster_settings,
    ):
        """Run forward rasterization.

        Args:
            ctx: Autograd context.
            means3D: 3D Gaussian means.
            means2D: 2D projected means.
            sh: Spherical harmonics coefficients.
            colors_precomp: Precomputed colors.
            flow_2d: 2D optical flow.
            opacities: Gaussian opacities.
            ts: Timestamps.
            scales: Spatial scales.
            scales_t: Temporal scales.
            rotations: Spatial rotations.
            rotations_r: Additional rotation parameters.
            cov3Ds_precomp: Precomputed 3D covariances.
            raster_settings: Rasterization settings container.

        Returns:
            Tuple of rasterized outputs.
        """
        # Arrange arguments in the exact order expected by the C++/CUDA extension.
        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            flow_2d,
            opacities,
            ts,
            scales,
            scales_t,
            rotations,
            rotations_r,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.sh_degree_t,
            raster_settings.campos,
            raster_settings.timestamp,
            raster_settings.time_duration,
            raster_settings.rot_4d,
            raster_settings.gaussian_dim,
            raster_settings.force_sh_3d,
            raster_settings.prefiltered,
            raster_settings.debug,
        )

        if raster_settings.debug:
            # Copy arguments before the CUDA call in case debugging requires a dump.
            cpu_args = tuple(
                item.cpu().clone() if isinstance(item, torch.Tensor) else item
                for item in args
            )
            try:
                (
                    num_rendered,
                    color,
                    flow,
                    depth,
                    T,
                    radii,
                    geomBuffer,
                    binningBuffer,
                    imgBuffer,
                    covs_com,
                    out_means3D,
                ) = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print(
                    "\nAn error occured in forward. Please forward "
                    "snapshot_fw.dump for debugging."
                )
                raise ex
        else:
            (
                num_rendered,
                color,
                flow,
                depth,
                T,
                radii,
                geomBuffer,
                binningBuffer,
                imgBuffer,
                covs_com,
                out_means3D,
            ) = _C.rasterize_gaussians(*args)

        # Save context for backward.
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(
            colors_precomp,
            means3D,
            out_means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii,
            sh,
            flow_2d,
            opacities,
            ts,
            scales_t,
            rotations_r,
            geomBuffer,
            binningBuffer,
            imgBuffer,
        )

        return color, radii, depth, 1 - T, flow, covs_com

    @staticmethod
    def backward(
        ctx,
        grad_out_color,
        grad_radii,
        grad_depth,
        grad_alpha,
        grad_flow,
        grad_covs_com,
    ):
        """Run backward rasterization and return input gradients."""
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        (
            colors_precomp,
            means3D,
            out_means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii,
            sh,
            flow_2d,
            opacities,
            ts,
            scales_t,
            rotations_r,
            geomBuffer,
            binningBuffer,
            imgBuffer,
        ) = ctx.saved_tensors

        # Arrange arguments in the exact order expected by the C++/CUDA backward.
        args = (
            raster_settings.bg,
            means3D,
            out_means3D,
            radii,
            colors_precomp,
            flow_2d,
            opacities,
            ts,
            scales,
            scales_t,
            rotations,
            rotations_r,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            grad_out_color,
            grad_depth,
            grad_alpha,
            grad_flow,
            sh,
            raster_settings.sh_degree,
            raster_settings.sh_degree_t,
            raster_settings.campos,
            raster_settings.timestamp,
            raster_settings.time_duration,
            raster_settings.rot_4d,
            raster_settings.gaussian_dim,
            raster_settings.force_sh_3d,
            geomBuffer,
            num_rendered,
            binningBuffer,
            imgBuffer,
            raster_settings.debug,
        )

        if raster_settings.debug:
            # Copy arguments before the CUDA call in case debugging requires a dump.
            cpu_args = tuple(
                item.cpu().clone() if isinstance(item, torch.Tensor) else item
                for item in args
            )
            try:
                (
                    grad_means2D,
                    grad_colors_precomp,
                    grad_opacities,
                    grad_means3D,
                    grad_cov3Ds_precomp,
                    grad_sh,
                    grad_flows,
                    grad_ts,
                    grad_scales,
                    grad_scales_t,
                    grad_rotations,
                    grad_rotations_r,
                ) = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print(
                    "\nAn error occured in backward. Writing "
                    "snapshot_bw.dump for debugging.\n"
                )
                raise ex
        else:
            (
                grad_means2D,
                grad_colors_precomp,
                grad_opacities,
                grad_means3D,
                grad_cov3Ds_precomp,
                grad_sh,
                grad_flows,
                grad_ts,
                grad_scales,
                grad_scales_t,
                grad_rotations,
                grad_rotations_r,
            ) = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_flows,
            grad_opacities,
            grad_ts,
            grad_scales,
            grad_scales_t,
            grad_rotations,
            grad_rotations_r,
            grad_cov3Ds_precomp,
            None,
        )
        return grads


class GaussianRasterizer(nn.Module):
    """Gaussian rasterizer wrapper around the C++/CUDA extension."""

    def __init__(self, raster_settings):
        """Initialize the rasterizer.

        Args:
            raster_settings: Rasterization settings container.
        """
        super().__init__()
        self.raster_settings = raster_settings

    def mark_visible(self, positions):
        """Mark points visible under the current camera frustum.

        Args:
            positions: 3D point positions.

        Returns:
            Boolean visibility mask.
        """
        # Mark visible points based on camera frustum culling.
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix,
            )

        return visible

    def forward(
        self,
        means3D,
        means2D,
        opacities,
        shs=None,
        colors_precomp=None,
        flow_2d=None,
        ts=None,
        scales=None,
        scales_t=None,
        rotations=None,
        rotations_r=None,
        cov3D_precomp=None,
    ):
        """Rasterize Gaussians with either SH colors or precomputed colors.

        Args:
            means3D: 3D Gaussian means.
            means2D: 2D projected Gaussian means.
            opacities: Gaussian opacities.
            shs: Optional spherical harmonics coefficients.
            colors_precomp: Optional precomputed colors.
            flow_2d: Optional 2D flow values.
            ts: Optional timestamps.
            scales: Optional spatial scales.
            scales_t: Optional temporal scales.
            rotations: Optional spatial rotations.
            rotations_r: Optional additional rotation parameters.
            cov3D_precomp: Optional precomputed 3D covariances.

        Returns:
            Rasterization outputs from the CUDA autograd function.
        """
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (
            shs is not None and colors_precomp is not None
        ):
            raise Exception(
                "Please provide excatly one of either SHs or precomputed colors!"
            )

        if (
            (scales is None or rotations is None) and cov3D_precomp is None
        ) or (
            (scales is not None or rotations is not None)
            and cov3D_precomp is not None
        ):
            raise Exception(
                "Please provide exactly one of either scale/rotation pair or "
                "precomputed 3D covariance!"
            )

        if (
            self.raster_settings.rot_4d
            and cov3D_precomp is None
            and (rotations_r is None or scales_t is None or ts is None)
        ):
            raise Exception(
                "Please provide exactly rotations_r and scales_t and ts if "
                "rot_4d and cov3D_precomp is None!"
            )

        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])
        if flow_2d is None:
            flow_2d = torch.Tensor([])
        if ts is None:
            ts = torch.Tensor([])
        if scales is None:
            scales = torch.Tensor([])
        if scales_t is None:
            scales_t = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if rotations_r is None:
            rotations_r = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke the C++/CUDA rasterization routine.
        return _RasterizeGaussians.apply(
            means3D,
            means2D,
            shs,
            colors_precomp,
            flow_2d,
            opacities,
            ts,
            scales,
            scales_t,
            rotations,
            rotations_r,
            cov3D_precomp,
            raster_settings,
        )
