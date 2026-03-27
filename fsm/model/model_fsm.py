#
#  Apache 2.0 License License
#  Copyright (c) 2026 Martin Ziqiao Ma
#
import copy
from abc import ABC, abstractmethod
import math

import torch
import torch.nn as nn

from .model_blocks import Block


def _init_weights(module):
    """Initialize module weights in place.

    Args:
        module: PyTorch module to initialize.
    """
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.RMSNorm, nn.LayerNorm)):
        module.reset_parameters()
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


class FSMBase(nn.Module, ABC):
    """Base class for Fast Spatial Memory models."""

    def __init__(self, config):

        super().__init__()

        self.current_step = None
        self.start_step = None
        self.max_step = None

        self.config = config
        self.config_backup = copy.deepcopy(config)

        self.patch_size = config.model.patch_size
        self.dim = config.model.dim

        block_config = config.model.get("block_config", ["ffn", "attn"])
        max_frames = config.model.max_frames

        self.pose_keys = config.model.plucker_repr
        self.posed_image_keys = self.pose_keys + ["normalized_image"]
        self.max_frames = max_frames

        self.input_dim = len(self.posed_image_keys) * 3
        self.input_layernorm = nn.LayerNorm(self.dim, bias=False)
        self.blocks = nn.ModuleList([
            Block(dim=self.dim, bias=False, block_config=block_config)
            for _ in range(config.model.num_layers)
        ])

    def apply_init_weights(self):
        """Initialize model weights."""

        # Apply the default module initialization first.
        self.apply(_init_weights)

        # Apply scaled initialization to residual projections, following GPT-2.
        block_config = self.config.model.get("block_config", ["ffn", "attn"])
        scale = 0.02 / math.sqrt(
            len(block_config) * self.config.model.num_layers
        )

        for param_name, param in self.named_parameters():
            if param_name.endswith("c_proj.weight"):
                torch.nn.init.normal_(param, mean=0.0, std=scale)

    @abstractmethod
    def set_curriculum(self, current_step, start_step=0, max_step=0):
        """Update the current training step.

        Subclasses may override this if step-dependent behavior is needed.

        Args:
            current_step: Current optimization step.
            start_step: Optional starting step.
            max_step: Optional maximum step.
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, input_data_dict, target_data_dict, info=None, skip_loss=False):
        """Run the forward pass.

        Args:
            input_data_dict: Input batch dictionary.
            target_data_dict: Target batch dictionary.
            info: Optional auxiliary information.

        Returns:
            Model-specific rendering outputs and loss metrics.
        """
        raise NotImplementedError
