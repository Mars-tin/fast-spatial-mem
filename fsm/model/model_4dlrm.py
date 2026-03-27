#
# Copyright (C) 2026, Adobe Research
# All rights reserved.
#
# This file is derived from author's self-reproduced 4D-LRM and TTT-LRM, 
# distributed under the terms of Adobe Research License.
#
# This file also incorporates material derived from code released by
# Ziwen Chen under the Apache 2.0 License.
#
# Modifications Copyright (c) 2026, Martin Ziqiao Ma
#
import copy
from einops import rearrange
from easydict import EasyDict as edict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses.manager import LossManager
from .utils_graphics import compute_rays
from .model_fsm import FSMBase
from .model_lacet import TTTOperator
from .model_4dgs import DeferredGaussianRender

class FSM4DLRM(FSMBase):

    def __init__(self, config):

        super().__init__(config)

        # Image tokenizer
        self.input_linear = nn.Linear((self.input_dim + 1)* (self.patch_size**2), self.dim, bias=False)
        
        # Determine d_gs based on temporal alignment
        self.scaling_modifier = config.model.gaussians.get("scaling_modifier", 1.0)
        self.d_gs = 3 + 1 + (config.model.gaussians.sh_degree + 1) ** 2 * 3 + 3 + 1 + 4 + 4 + 1
        if config.model.get("hard_tempalign", False):
            self.d_gs -= 1

        # GS token decoder
        self.image_token_decoder = nn.Sequential(
            nn.LayerNorm(self.dim, bias=False),
            nn.Linear(self.dim, (self.patch_size**2) * self.d_gs, bias=False),
        )

        # Loss manager
        self.loss_mgr = LRMLossManager(config)

        self.apply_init_weights()
    
    def set_curriculum(self, current_step, start_step=0, max_step=0):

        self.current_step = current_step
        self.start_step = start_step
        self.max_step = max_step

        plan_to_modify_config = (
            self.config.training.get("pointsdist_warmup_steps", 0) > 0
            and self.current_step < self.config.training.get("pointsdist_warmup_steps", 0)
        ) or (
            self.config.training.get("l2_warmup_steps", 0) > 0
            and self.current_step < self.config.training.get("l2_warmup_steps", 0)
        )
        if plan_to_modify_config:
            # always use the self.config_backup as the starting point for modification
            self.config = copy.deepcopy(self.config_backup)

            if self.config.training.get("pointsdist_warmup_steps", 0) > 0:
                if self.current_step < self.config.training.get("pointsdist_warmup_steps", 0):
                    self.config.training.l2_loss_weight = 0.0
                    self.config.training.perceptual_loss_weight = 0.0
                    self.config.training.pointsdist_loss_weight = 0.1
                    self.config.model.clip_xyz = (
                        False  # turn off xyz clipping for warmup
                    )

            if self.config.training.get("l2_warmup_steps", 0) > 0:
                if self.current_step < self.config.training.get("l2_warmup_steps", 0):
                    self.config.training.perceptual_loss_weight = 0.0
                    self.config.training.lpips_loss_weight = 0.0
        else:
            self.config = self.config_backup

        self.loss_mgr.config = self.config
        
    def forward(self, input_data_dict, target_data_dict, info=None, skip_loss=False):

        # Do not autocast during the data processing
        with torch.autocast(device_type="cuda", enabled=False), torch.no_grad():
            batch_size, num_input_views, _, height, width = input_data_dict["image"].size()
            num_target_views = target_data_dict["c2w"].size(1)
            input_frame_time = input_data_dict['frame_time']
            output_frame_time = target_data_dict['frame_time']
            
            for data_dict in [input_data_dict, target_data_dict]:
                fxfycxcy = data_dict["fxfycxcy"]
                c2w = data_dict["c2w"]

                data_dict["ray_o"], data_dict["ray_d"] = compute_rays(fxfycxcy, c2w, height, width)
                data_dict["o_cross_d"] = torch.cross(data_dict["ray_o"], data_dict["ray_d"], dim=2)
                o_dot_d = torch.sum(-data_dict["ray_o"] * data_dict["ray_d"], dim=2, keepdim=True)
                data_dict["nearest_pts"] = data_dict["ray_o"] + o_dot_d * data_dict["ray_d"]
                data_dict["pose_only"] = torch.concat(
                    [data_dict[key] for key in self.pose_keys], dim=2
                )
                
                if "image" in data_dict:
                    data_dict["normalized_image"] = data_dict["image"] * 2.0 - 1.0

                    # Compile the information for posed-image input, and pose-only input.
                    data_dict["posed_image"] = torch.concat(
                        [data_dict[key] for key in self.posed_image_keys], dim=2
                    )

            add_target_views = self.config.model.get("add_target_views", False)
            add_virtual_views = self.config.model.get("add_virtual_views", True)
            
            num_img_tokens = height * width // (self.patch_size**2)
            num_input_tokens = num_input_views * num_img_tokens
        
            # LVSM style (w/ Virtual Views = Target Views)
            if add_target_views:

                num_target_tokens = num_target_views * num_img_tokens

                transformer_input = input_data_dict["image"].new_zeros(
                    batch_size, num_input_views + num_target_views, self.input_dim, height, width
                )
                transformer_input[:, :num_input_views, :, :, :] = input_data_dict["posed_image"]
                pose_only_dim = target_data_dict["pose_only"].size(2)
                transformer_input[:, num_input_views:, :pose_only_dim, :, :] = target_data_dict["pose_only"]

                input_frame_time = input_frame_time.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(batch_size, num_input_views, 1, height, width)  # Add frame time
                output_frame_time = output_frame_time.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(batch_size, num_target_views, 1, height, width)  # Add frame time
                input_frame_time = input_frame_time / (self.max_frames - 1) * 2.0 - 1.0
                output_frame_time = output_frame_time / (self.max_frames - 1) * 2.0 - 1.0
                frame_time = torch.cat([input_frame_time, output_frame_time], dim=1)  # Concatenate frame time

                transformer_input = torch.cat([transformer_input, frame_time], dim=2)  # Concatenate frame time to the input

            # tttLRM style (w/ Virtual Views = Input Views)
            elif add_virtual_views:
                
                # tttLRM adopts the input as virtual views
                num_target_tokens = num_input_views * num_img_tokens

                posed = input_data_dict["posed_image"]                      # (B, V, C, H, W)
                transformer_input = torch.cat([posed, posed], dim=1)        # (B, 2V, C, H, W)

                input_frame_time = input_frame_time.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(batch_size, num_input_views, 1, height, width)  # Add frame time
                input_frame_time = input_frame_time / (self.max_frames - 1) * 2.0 - 1.0
                frame_time = torch.cat([input_frame_time, input_frame_time], dim=1)  # Concatenate frame time

                transformer_input = torch.cat([transformer_input, frame_time], dim=2)  # Concatenate frame time to the input

            # Vanilla GSLRM style (w/o Virtual Views)
            else:
                
                num_target_tokens = 0

                transformer_input = input_data_dict["posed_image"]
                frame_time = input_frame_time.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(batch_size, num_input_views, 1, height, width)  # Add frame time
                frame_time = frame_time / (self.max_frames - 1) * 2.0 - 1.0

                transformer_input = torch.cat([transformer_input, frame_time], dim=2)  # Concatenate frame time to the input

        # Running the model
        ttt_op_order = [
            TTTOperator(start=0, end=num_input_tokens, update=True, apply=False),
            TTTOperator(start=0, end=num_input_tokens + num_target_tokens, update=False, apply=True),
        ]
        if info is None or not info["ewc_enabled"]:
            info = {"ttt_op_order": ttt_op_order, "num_img_tokens": num_img_tokens}
        else:
            info.update({"ttt_op_order": ttt_op_order, "num_img_tokens": num_img_tokens})

        x = rearrange(
            transformer_input,
            "b v c (hh ph) (ww pw) -> b (v hh ww) (ph pw c)",
            ph=self.patch_size,
            pw=self.patch_size,
        )
        x = self.input_linear(x)
        x = self.input_layernorm(x)
        for block in self.blocks:
            x, _ = block(x, info)

        # Extract input tokens
        if add_target_views or add_virtual_views:
            input_x = x[:, -num_target_tokens:]
        else:
            input_x = x[:, :num_input_tokens]

        # Decode Pixel-Aligned Gaussians
        img_aligned_gaussians = self.image_token_decoder(input_x)
        img_aligned_gaussians = img_aligned_gaussians.reshape(batch_size, -1, self.d_gs)
        
        # Split into physical attributes
        # xyz here is still a latent value, requiring depth unprojection in the next step
        xyz_raw, t, features, scaling, scaling_t, rotation, rotation_r, opacity = self.to_gs(img_aligned_gaussians)

        # Extract ray parameters from input_data_dict
        in_ray_o = input_data_dict["ray_o"] # [b, v_in, 3, h, w]
        in_ray_d = input_data_dict["ray_d"] # [b, v_in, 3, h, w]

        # Reshape xyz_raw back to image space for ray computation
        img_aligned_xyz = rearrange(
            xyz_raw,
            "b (v h w ph pw) c -> b v c (h ph) (w pw)",
            v=num_input_views,
            h=height // self.patch_size,
            w=width // self.patch_size,
            ph=self.patch_size,
            pw=self.patch_size,
        )

        # Convert the Transformer-predicted xyz_raw to depth
        # [TODO] add these magic numbers to config
        # [TODO] the near/far hear are offsets for LRM training,
        #        in model_4dgs.py, znear is actually default to 0.01.
        #        this honestly does not matter much as long as the depth is properly scaled,
        #        yet we should unify these numbers for consistency later.
        # Near plane: object level -> 0.1, scene level -> 0.0
        near = self.config.model.gaussians.get("near_plane", 0.0)
        # Far plane: object level -> 4.5, scene level -> 100.0
        far = self.config.model.gaussians.get("far_plane", 100.0)
        # Depth scale: object level -> 2.0, scene level -> 1.0
        scale_d = self.config.model.gaussians.get("depth_scale", 1.0)

        if self.config.model.get("hard_pixelalign", False):
            depth_preact_bias = self.config.model.get("depth_preact_bias", 0.0)
            depth = torch.sigmoid(img_aligned_xyz.mean(dim=2, keepdim=True) + depth_preact_bias)
        else:
            depth = img_aligned_xyz.mean(dim=2, keepdim=True)
        depth = scale_d * depth * (far - near) + near

        # Core geometric constraint: Absolute Coordinate = Camera Origin + Depth * Ray Direction
        xyz_world = in_ray_o + depth * in_ray_d

        # Set clip_xyz = False for scene
        if self.config.model.get("clip_xyz", False):
            xyz_world = xyz_world.clamp(-1.0, 1.0)

        # Reshape the generated absolute 3D coordinates back to point cloud shape [b, n_gaussians, 3]
        xyz = rearrange(
            xyz_world,
            "b v c (h ph) (w pw) -> b (v h w ph pw) c",
            ph=self.patch_size,
            pw=self.patch_size,
        )

        # Deferred rendering
        assert self.max_frames > 1, "max_frames must be greater than 1 for temporal encoding"
        output_frame_time = target_data_dict["frame_time"] / (self.max_frames - 1) * 2.0 - 1.0
        renderings = DeferredGaussianRender.apply(
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
            target_data_dict["c2w"],
            target_data_dict["fxfycxcy"],
            output_frame_time,
            self.scaling_modifier,
            (self.config.inference or self.config.get("evaluation", False)),
            self.config.model.get("t_threshold", 0.05)
        )
        
        losses = None
        if not skip_loss:
            # Manage losses and schedule
            input_edict = edict(input_data_dict) 
            losses = self.loss_mgr(
                renderings, 
                target_data_dict["image"], 
                xyz_world, 
                input_edict
            )

        return renderings, losses

    def to_gs(self, gaussians):
        """
        gaussians: [b, n_gs_all, d_gs]
        n_gs_all = n_gaussians + n_gs_pixel (tv * n_pixels)
        """
        if self.config.model.get("hard_tempalign", False):
            xyz, features, scaling, scaling_t, rotation, rotation_r, opacity = gaussians.split(
                [3, (self.config.model.gaussians.sh_degree + 1) ** 2 * 3, 3, 1, 4, 4, 1], dim=2
            )
            t = None
        elif self.config.model.get("init_from_gslrm", False):
            xyz, features, scaling, rotation, opacity, t, scaling_t, rotation_r = gaussians.split(
                [3, (self.config.model.gaussians.sh_degree + 1) ** 2 * 3, 3, 4, 1, 1, 1, 4], dim=2
            )
        else:
            xyz, t, features, scaling, scaling_t, rotation, rotation_r, opacity = gaussians.split(
                [3, 1, (self.config.model.gaussians.sh_degree + 1) ** 2 * 3, 3, 1, 4, 4, 1], dim=2
            )
        features = features.reshape(
            features.size(0),
            features.size(1),
            (self.config.model.gaussians.sh_degree + 1) ** 2,
            3,
        )

        # 4DGS Parameterization Biases and Clips
        # [TODO] add these magic numbers to config
        # Opacity bias: object level -> 2.0, scene level -> 0.0
        opac_bias = self.config.model.gaussians.get("opacity_bias", 0.0)
        # Scaling bias: object level -> 2.3, scene level -> 0.0
        scale_bias = self.config.model.gaussians.get("scaling_bias", 0.0)
        # Temporal scaling bias: object level -> 2.3, scene level -> 0.0
        scale_t_bias = self.config.model.gaussians.get("temporal_scaling_bias", 0.0)
        # Scaling clip: object level -> -1.20, scene level -> 10.0
        scale_clip = self.config.model.gaussians.get("scaling_clip", 10.0)
        # Temporal scaling clip: object level -> -1.20, scene level -> 0.0
        scale_t_clip = self.config.model.gaussians.get("temporal_scaling_clip", 0.0)
        
        opacity = opacity - opac_bias
        scaling = (scaling - scale_bias).clamp(max=scale_clip)
        if self.d_gs != 3:
            scaling_t = (scaling_t - scale_t_bias).clamp(max=scale_t_clip)

        # xyz       =   [b, n_gs_all, 3]
        # features  =   [b, n_gs_all, (sh+1)^2, 3], default: sh=0
        # scaling   =   [b, n_gs_all, 3]
        # rotation  =   [b, n_gs_all, 4]
        # opacity   =   [b, n_gs_all, 1]
        return xyz, t, features, scaling, scaling_t, rotation, rotation_r, opacity


class LRMLossManager(LossManager):
    """Loss manager for LRM-style rendering models."""

    def __init__(self, config):
        """Initialize the loss manager.

        Args:
            config: Configuration object containing training loss weights.
        """
        super().__init__(config)

    def forward(
        self,
        rendering,
        target,
        input,
        img_aligned_xyz
    ):
        """Compute training losses for rendered images.

        Args:
            rendering: Rendered images of shape [b, v, 3, h, w] in range (0, 1).
            target: Target images of shape [b, v, 3, h, w] or [b, v, 4, h, w]
                in range (0, 1). If 4 channels are provided, the last channel is
                treated as a mask.
            img_aligned_xyz: Aligned 3D coordinates.
            input: Input batch containing ray origins and directions.

        Returns:
            edict: Dictionary-like object containing the total loss and all
            individual loss terms.
        """
        b, v, _, h, w = rendering.size()
        rendering = rendering.reshape(b * v, -1, h, w)
        target = target.reshape(b * v, -1, h, w)

        # L2 reconstruction loss.
        l2_loss = torch.tensor(1e-8).to(rendering.device)
        l2_loss = F.mse_loss(rendering, target)

        # PSNR computed from the batch-level L2 loss. This is not identical to
        # averaging per-image PSNR, but is still useful for monitoring.
        psnr = -10.0 * torch.log10(l2_loss.clamp(min=1e-8))

        # LPIPS loss.
        lpips_loss = torch.tensor(0.0).to(l2_loss.device)
        if self.config.training.lpips_loss_weight > 0.0:
            lpips_loss = self.lpips_loss_module(
                rendering,
                target,
                normalize=True,
            ).mean()

        # Perceptual loss.
        perceptual_loss = torch.tensor(0.0).to(l2_loss.device)
        if self.config.training.perceptual_loss_weight > 0.0:
            perceptual_loss = self.perceptual_loss_module(rendering, target)

        # SSIM loss.
        ssim_loss = torch.tensor(0.0).to(l2_loss.device)
        if self.config.training.ssim_loss_weight > 0.0:
            ssim_loss = self.ssim_loss_module(rendering, target)

        # Pixel-alignment loss.
        # [TODO] pixelalign_loss is not tested for public release
        mask = None
        if target.size(1) == 4:
            target, mask = target.split([3, 1], dim=1)

        pixelalign_loss = torch.tensor(0.0).to(l2_loss.device)
        if self.config.training.pixelalign_loss_weight > 0.0:
            xyz_vec = img_aligned_xyz - input.ray_o
            ortho_vec = (
                xyz_vec
                - torch.sum(
                    xyz_vec.detach() * input.ray_d,
                    dim=2,
                    keepdim=True,
                )
                * input.ray_d
            )

            if self.config.training.get("pixelalign_loss_weight", 0.0) > 0.0:
                assert mask is not None, "mask is None"
                # [b * v, 1, h, w] -> [b, v, 1, h, w]
                mask = mask.view(b, v, 1, h, w)
                ortho_vec = ortho_vec * mask

            pixelalign_loss = torch.mean(ortho_vec.norm(dim=2, p=2))

        # Scene-level point distance loss.
        # [TODO] support object-level pointsdist_loss later
        pointsdist_loss = torch.tensor(0.0).to(l2_loss.device)
        if self.config.training.pointsdist_loss_weight > 0.0:
            trgt_mean_dist = torch.norm(
                input.ray_o,
                dim=2,
                p=2,
                keepdim=True,
            )  # [b, v, 1, h, w]

            dist = (img_aligned_xyz - input.ray_o).norm(
                dim=2,
                p=2,
                keepdim=True,
            )  # [b, v, 1, h, w]

            dist_detach = dist.detach()
            dist_mean = dist_detach.mean(dim=(2, 3, 4), keepdim=True)
            dist_std = dist_detach.std(dim=(2, 3, 4), keepdim=True)

            trgt_std_dist = 0.5
            trgt_dist = (
                (dist_detach - dist_mean) / (dist_std + 1e-8) * trgt_std_dist
                + trgt_mean_dist
            )

            pointsdist_loss = torch.mean((dist - trgt_dist) ** 2)

        loss = (
            self.config.training.get("l2_loss_weight", 1.0) * l2_loss
            + self.config.training.get("lpips_loss_weight", 0.0) * lpips_loss
            + self.config.training.get("perceptual_loss_weight", 0.5) * perceptual_loss
            + self.config.training.get("ssim_loss_weight", 0.0) * ssim_loss
            + self.config.training.get("pixelalign_loss_weight", 0.0) * pixelalign_loss
            + self.config.training.get("pointsdist_loss_weight", 0.0) * pointsdist_loss
        )

        loss_metrics = edict(
            loss=loss,
            l2_loss=l2_loss,
            psnr=psnr,
            lpips_loss=lpips_loss,
            perceptual_loss=perceptual_loss,
            ssim_loss=ssim_loss,
            pixelalign_loss=pixelalign_loss,
            pointsdist_loss=pointsdist_loss,
        )
        return loss_metrics
