"""Loss construction for multi-class segmentation.

Networks emit raw logits of shape (B, C, H, W, D); labels are integer maps of
shape (B, 1, H, W, D). Both losses handle the one-hot conversion and softmax
internally.
"""

from __future__ import annotations

import torch.nn as nn
from monai.losses import DiceCELoss, DiceFocalLoss


def build_loss(
    name: str = "dice_ce",
    include_background: bool = False,
    lambda_dice: float = 1.0,
    lambda_ce: float = 1.0,
) -> nn.Module:
    """Return a MONAI loss module.

    Args:
        name: ``"dice_ce"`` or ``"dice_focal"``.
        include_background: whether the background class contributes to the loss.
            Kept False by default — the foreground (thin leaflet/annulus) classes
            are what we care about and background dominates the volume.
    """
    common = {
        "include_background": include_background,
        "to_onehot_y": True,
        "softmax": True,
    }
    if name == "dice_ce":
        return DiceCELoss(lambda_dice=lambda_dice, lambda_ce=lambda_ce, **common)
    if name == "dice_focal":
        return DiceFocalLoss(lambda_dice=lambda_dice, lambda_focal=lambda_ce, **common)
    raise ValueError(f"Unknown loss {name!r} (use 'dice_ce' or 'dice_focal')")
