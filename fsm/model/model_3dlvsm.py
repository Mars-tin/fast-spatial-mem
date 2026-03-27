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
import copy
from einops import rearrange
from easydict import EasyDict as edict

import torch
import torch.nn as nn
from torch.nn import functional as F

from .utils_graphics import compute_rays
from .model_fsm import FSMBase
from .model_lacet import TTTOperator
from .model_4dlvsm import LVSMLossManager

class FSM3DLVSM(FSMBase):

    def __init__(self, config):
        super().__init__(config)
        self.input_linear = nn.Linear(self.input_dim * (self.patch_size**2), self.dim, bias=False)
        self.image_token_decoder = nn.Sequential(
            nn.LayerNorm(self.dim, bias=False),
            nn.Linear(self.dim, (self.patch_size**2) * 3, bias=False),
            nn.Sigmoid()
        )
        self.loss_mgr = LVSMLossManager(config)
        self.apply_init_weights()

    def set_curriculum(self, current_step, start_step=0, max_step=0):

        self.current_step = current_step
        self.start_step = start_step
        self.max_step = max_step

        plan_to_modify_config = (
            self.config.training.get("l2_warmup_steps", 0) > 0
            and self.current_step < self.config.training.get("l2_warmup_steps", 0)
        )
        if plan_to_modify_config:
            # always use the self.config_backup as the starting point for modification
            self.config = copy.deepcopy(self.config_backup)

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
            batch_size, num_input_views, _, h, w = input_data_dict["image"].size()
            num_target_views = target_data_dict["c2w"].size(1)
            
            for data_dict in [input_data_dict, target_data_dict]:
                fxfycxcy = data_dict["fxfycxcy"]
                c2w = data_dict["c2w"]

                data_dict["ray_o"], data_dict["ray_d"] = compute_rays(fxfycxcy, c2w, h, w)
                data_dict["o_cross_d"] = torch.cross(data_dict["ray_o"], data_dict["ray_d"], dim=2)
                data_dict["pose_only"] = torch.concat(
                    [data_dict[key] for key in self.pose_keys], dim=2
                )
                
                if "image" in data_dict:
                    data_dict["normalized_image"] = data_dict["image"] * 2.0 - 1.0

                    # Compile the information for posed-image input, and pose-only input.
                    data_dict["posed_image"] = torch.concat(
                        [data_dict[key] for key in self.posed_image_keys], dim=2
                    )
            
            transformer_input = input_data_dict["image"].new_zeros(
                batch_size, num_input_views + num_target_views, self.input_dim, h, w
            )

            transformer_input[:, :num_input_views, :, :, :] = input_data_dict["posed_image"]
            pose_only_dim = target_data_dict["pose_only"].size(2)
            transformer_input[:, num_input_views:, :pose_only_dim, :, :] = target_data_dict["pose_only"]

        # Running the model
        num_img_tokens = h * w // (self.patch_size**2)
        num_input_tokens = num_input_views * num_img_tokens
        num_target_tokens = num_target_views * num_img_tokens
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
        
        target_x = x[:, -num_target_tokens:]
        target_x = self.image_token_decoder(target_x)
        renderings = rearrange(
            target_x,
            "b (v hh ww) (ph pw c) -> b v c (hh ph) (ww pw)",
            v=num_target_views,
            hh=h // self.patch_size,
            ww=w // self.patch_size,
            ph=self.patch_size,
            pw=self.patch_size,
            c=3,
        )
        
        losses = None
        if not skip_loss:
            # Manage losses and schedule
            losses = self.loss_mgr(
                renderings, 
                target_data_dict["image"]
            )

        return renderings, losses
