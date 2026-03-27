
#
#  Apache 2.0 License License
#  Copyright (c) 2026 Martin Ziqiao Ma
#
from abc import ABC, abstractmethod
import torch.nn as nn
from .perceptual_loss import PerceptualLoss
from .ssim_loss import SSIMLoss

class LossManager(nn.Module, ABC):
    """Abstract base class for managing training losses."""

    def __init__(self, config):
        """Initialize optional loss modules based on the training config.

        Args:
            config: Configuration object containing training loss weights.
        """
        super().__init__()
        self.config = config

        if self.config.training.get("lpips_loss_weight", 0.0) > 0.0:
            from lpips import LPIPS
            self.lpips_loss_module = LPIPS(net="vgg")
            self.lpips_loss_module.eval()
            for param in self.lpips_loss_module.parameters():
                param.requires_grad = False

        if self.config.training.get("perceptual_loss_weight", 0.0) > 0.0:
            self.perceptual_loss_module = PerceptualLoss()
            self.perceptual_loss_module.eval()
            for param in self.perceptual_loss_module.parameters():
                param.requires_grad = False

        if self.config.training.get("ssim_loss_weight", 0.0) > 0.0:
            self.ssim_loss_module = SSIMLoss()
            self.ssim_loss_module.eval()
            for param in self.ssim_loss_module.parameters():
                param.requires_grad = False

    @abstractmethod
    def forward(
        self,
        rendering,
        target,
        input=None,
        img_aligned_xyz=None
    ):
        """Compute losses for a model output.

        Args:
            rendering: Rendered model output.
            target: Ground-truth target.
            img_aligned_xyz: Optional aligned 3D coordinates.
            input: Optional input batch or metadata.

        Returns:
            Model-specific loss outputs.
        """
        raise NotImplementedError
