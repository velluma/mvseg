"""Loss construction for multi-class segmentation.

Networks emit raw logits of shape (B, C, H, W, D); labels are integer maps of
shape (B, 1, H, W, D). The base losses handle one-hot conversion and softmax
internally.

We also provide a **clDice** (centerline-Dice) term for topology preservation of
thin, ring-/tube-shaped structures (e.g. the aortic-valve annulus, label 4),
which tend to break into disconnected pieces under a pure overlap loss. clDice
[Shit et al., CVPR 2021] measures agreement between the *soft skeletons* of the
prediction and the ground truth, so it explicitly rewards keeping the centerline
connected. It is added on top of the voxel-overlap loss for selected classes.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss, DiceFocalLoss

# --------------------------------------------------------------------------- #
# Soft morphology + clDice (3D)
# --------------------------------------------------------------------------- #


def _soft_erode(img: torch.Tensor) -> torch.Tensor:
    """Grayscale erosion via separable min-pooling (min = -maxpool(-x))."""
    p1 = -F.max_pool3d(-img, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))
    p2 = -F.max_pool3d(-img, kernel_size=(1, 3, 1), stride=1, padding=(0, 1, 0))
    p3 = -F.max_pool3d(-img, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1))
    return torch.min(torch.min(p1, p2), p3)


def _soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool3d(img, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))


def _soft_open(img: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(img))


def soft_skeletonize(img: torch.Tensor, iters: int = 10) -> torch.Tensor:
    """Differentiable soft skeleton of a probability map in ``[0, 1]``.

    Iteratively erodes the mask and accumulates the "boundary residual"
    (``relu(img - open(img))``). ``iters`` should be >= the expected half-radius
    (in voxels) of the thickest structure so the skeleton fully forms.

    Args:
        img: ``(B, 1, H, W, D)`` tensor in ``[0, 1]``.
    """
    img1 = _soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iters):
        img = _soft_erode(img)
        img1 = _soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    iters: int = 10,
    smooth: float = 1.0,
) -> torch.Tensor:
    """1 - clDice for a single (binary) channel.

    Args:
        pred:   ``(B, 1, H, W, D)`` predicted foreground probability in ``[0, 1]``.
        target: ``(B, 1, H, W, D)`` binary ground-truth mask in ``{0, 1}``.
    """
    skel_pred = soft_skeletonize(pred, iters)
    skel_true = soft_skeletonize(target, iters)
    # Topology precision: how much of the predicted skeleton lies inside the GT.
    tprec = (torch.sum(skel_pred * target) + smooth) / (torch.sum(skel_pred) + smooth)
    # Topology sensitivity: how much of the GT skeleton is covered by the prediction.
    tsens = (torch.sum(skel_true * pred) + smooth) / (torch.sum(skel_true) + smooth)
    cldice = 2.0 * tprec * tsens / (tprec + tsens)
    return 1.0 - cldice


class ClDiceAugmentedLoss(nn.Module):
    """Wrap a base (multi-class) loss and add a clDice term on selected classes.

    ``total = base_loss(logits, target) + lambda_cldice * mean_c clDice_c``

    The clDice term is applied per class ``c`` in ``cldice_classes`` on the
    softmax foreground probability of that channel vs. its binary GT mask. Default
    targets the aortic-valve annulus (index 4), the thinnest ring-like structure.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        cldice_classes: Sequence[int] = (4,),
        lambda_cldice: float = 0.5,
        cldice_iters: int = 10,
        cldice_smooth: float = 1.0,
    ):
        super().__init__()
        if not cldice_classes:
            raise ValueError("cldice_classes must list >= 1 class index")
        self.base_loss = base_loss
        self.cldice_classes = tuple(int(c) for c in cldice_classes)
        self.lambda_cldice = float(lambda_cldice)
        self.cldice_iters = int(cldice_iters)
        self.cldice_smooth = float(cldice_smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        base = self.base_loss(logits, target)
        probs = torch.softmax(logits, dim=1)
        # target is (B, 1, H, W, D) integer -> squeeze the channel to (B, H, W, D)
        tgt = target[:, 0] if target.dim() == probs.dim() else target
        terms = []
        for c in self.cldice_classes:
            pred_c = probs[:, c : c + 1]
            gt_c = (tgt == c).unsqueeze(1).to(probs.dtype)
            terms.append(
                soft_cldice_loss(pred_c, gt_c, iters=self.cldice_iters, smooth=self.cldice_smooth)
            )
        cldice = torch.stack(terms).mean()
        return base + self.lambda_cldice * cldice


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #

_CLDICE_SUFFIX = "_cldice"


def build_loss(
    name: str = "dice_ce",
    include_background: bool = False,
    lambda_dice: float = 1.0,
    lambda_ce: float = 1.0,
    cldice_classes: Sequence[int] = (4,),
    lambda_cldice: float = 0.5,
    cldice_iters: int = 10,
    cldice_smooth: float = 1.0,
) -> nn.Module:
    """Return a loss module.

    Args:
        name: base loss ``"dice_ce"`` / ``"dice_focal"``, optionally with a
            ``"_cldice"`` suffix (``"dice_ce_cldice"``) to add the topology term.
        include_background: whether background contributes to the base loss.
            Kept False by default — the thin foreground classes are what matter
            and background dominates the volume.
        cldice_classes: class indices the clDice term is applied to (default the
            aortic-valve annulus, 4). Only used when ``name`` ends with ``_cldice``.
        lambda_cldice: weight of the clDice term relative to the base loss.
        cldice_iters: soft-skeletonization iterations (>= thickest half-radius).
    """
    use_cldice = name.endswith(_CLDICE_SUFFIX)
    base_name = name[: -len(_CLDICE_SUFFIX)] if use_cldice else name

    common = {
        "include_background": include_background,
        "to_onehot_y": True,
        "softmax": True,
    }
    if base_name == "dice_ce":
        base: nn.Module = DiceCELoss(lambda_dice=lambda_dice, lambda_ce=lambda_ce, **common)
    elif base_name == "dice_focal":
        base = DiceFocalLoss(lambda_dice=lambda_dice, lambda_focal=lambda_ce, **common)
    else:
        raise ValueError(
            f"Unknown loss {name!r} (use 'dice_ce', 'dice_focal', or a '_cldice' variant)"
        )

    if not use_cldice:
        return base
    return ClDiceAugmentedLoss(
        base,
        cldice_classes=cldice_classes,
        lambda_cldice=lambda_cldice,
        cldice_iters=cldice_iters,
        cldice_smooth=cldice_smooth,
    )
