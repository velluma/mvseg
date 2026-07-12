"""Segmentation metrics: per-class Dice and 95% Hausdorff distance.

We report foreground classes individually (anterior/posterior leaflet, MV/AV
annulus) plus a foreground mean, since a single averaged score hides failures on
the small, clinically important structures.
"""

from __future__ import annotations

import torch
from monai.data import decollate_batch
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete, Compose

from mvseg import FG_CLASS_NAMES


class SegMetrics:
    """Accumulates Dice/HD95 over a validation or test epoch.

    Usage:
        m = SegMetrics(num_classes=5)
        m.update(logits, labels)     # per step
        results = m.aggregate()      # {"dice_mean_fg": ..., "dice/anterior_leaflet": ...}
        m.reset()
    """

    def __init__(self, num_classes: int = 5, compute_hd95: bool = True):
        self.num_classes = num_classes
        self.compute_hd95 = compute_hd95
        self.dice = DiceMetric(include_background=False, reduction="mean_batch")
        self.hd95 = (
            HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean_batch")
            if compute_hd95
            else None
        )
        self._post_pred = Compose([AsDiscrete(argmax=True, to_onehot=num_classes)])
        self._post_label = Compose([AsDiscrete(to_onehot=num_classes)])

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        preds = [self._post_pred(p) for p in decollate_batch(logits)]
        gts = [self._post_label(g) for g in decollate_batch(labels)]
        self.dice(y_pred=preds, y=gts)
        if self.hd95 is not None:
            self.hd95(y_pred=preds, y=gts)

    def aggregate(self) -> dict[str, float]:
        out: dict[str, float] = {}
        dice_per_class = self.dice.aggregate()  # shape (num_fg_classes,)
        dice_vals = _to_list(dice_per_class)
        for name, val in zip(FG_CLASS_NAMES, dice_vals, strict=False):
            out[f"dice/{name}"] = val
        out["dice_mean_fg"] = float(sum(dice_vals) / len(dice_vals)) if dice_vals else 0.0

        if self.hd95 is not None:
            hd_vals = _to_list(self.hd95.aggregate())
            for name, val in zip(FG_CLASS_NAMES, hd_vals, strict=False):
                out[f"hd95/{name}"] = val
            finite = [v for v in hd_vals if v == v and v != float("inf")]
            out["hd95_mean_fg"] = float(sum(finite) / len(finite)) if finite else float("nan")
        return out

    def reset(self) -> None:
        self.dice.reset()
        if self.hd95 is not None:
            self.hd95.reset()


def _to_list(t) -> list[float]:
    if isinstance(t, torch.Tensor):
        return [float(x) for x in t.flatten().tolist()]
    return [float(t)]
