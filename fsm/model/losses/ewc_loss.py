
#
#  Apache 2.0 License License
#  Copyright (c) 2026 Martin Ziqiao Ma
#
import torch


def ewc_training_loss(blocks, ewc_buffers):
    """Compute the optional training-time EWC loss across all fast blocks.

    The loss is defined as the mean of F * (w - w*)^2 for each fast-weight
    parameter group, averaged within each block and summed across blocks.

    Args:
        blocks: Iterable of (name, block) pairs.
        ewc_buffers: Mapping from block name to stored EWC statistics.

    Returns:
        torch.Tensor: Scalar EWC regularization loss.
    """
    total = None

    for name, blk in blocks:
        buf = ewc_buffers[name]

        # Detach all stored reference weights and Fisher terms. Only the
        # current block parameters should receive gradients.
        w0, w1, w2 = blk.w0, blk.w1, blk.w2
        w0s = buf["w0_star"].detach()
        F0 = buf["F0"].detach()
        w1s = buf["w1_star"].detach()
        F1 = buf["F1"].detach()
        w2s = buf["w2_star"].detach()
        F2 = buf["F2"].detach()

        term = (
            (F0 * (w0 - w0s).pow(2)).mean()
            + (F1 * (w1 - w1s).pow(2)).mean()
            + (F2 * (w2 - w2s).pow(2)).mean()
        ) / 3.0

        total = term if total is None else total + term

    if total is None:
        return torch.tensor(0.0)

    return total
