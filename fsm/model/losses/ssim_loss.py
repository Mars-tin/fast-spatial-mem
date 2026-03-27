#
# Copyright (C) 2026, Adobe Research
# All rights reserved.
#
# This file is derived from author's self-reproduced 4D-LRM, 
# distributed under the terms of Adobe Research License.
#
# Modifications Copyright (c) 2026, Martin Ziqiao Ma
#
import torch.nn as nn
from pytorch_msssim import SSIM


class SSIMLoss(nn.Module):
    """Structural similarity loss defined as 1 - SSIM."""

    def __init__(self, data_range=1.0):
        """Initialize the SSIM loss module.

        Args:
            data_range: Value range of the input images.
        """
        super().__init__()
        self.ssim_module = SSIM(
            win_size=11,
            win_sigma=1.5,
            data_range=data_range,
            size_average=True,
            channel=3,
        )

    def forward(self, x, y):
        """Compute SSIM loss.

        Args:
            x: Input image tensor of shape (N, C, H, W).
            y: Target image tensor of shape (N, C, H, W).

        Returns:
            SSIM loss value.
        """
        return 1 - self.ssim_module(x, y)
