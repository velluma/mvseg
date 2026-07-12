"""Metric correctness tests."""

from __future__ import annotations

import torch

from mvseg import FG_CLASS_NAMES
from mvseg.metrics import SegMetrics


def _one_hot_logits(label: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Build logits that argmax exactly to ``label`` (B,1,...) -> (B,C,...)."""
    b = label.shape[0]
    spatial = label.shape[2:]
    logits = torch.full((b, num_classes, *spatial), -10.0)
    for c in range(num_classes):
        logits[:, c][label[:, 0] == c] = 10.0
    return logits


def test_perfect_prediction_dice_is_one(num_classes):
    label = torch.zeros(1, 1, 16, 16, 16, dtype=torch.long)
    # place one voxel of each foreground class
    for c in range(1, num_classes):
        label[0, 0, c, c, c] = c
    logits = _one_hot_logits(label, num_classes)

    m = SegMetrics(num_classes=num_classes, compute_hd95=False)
    m.update(logits, label)
    res = m.aggregate()

    assert abs(res["dice_mean_fg"] - 1.0) < 1e-5
    for name in FG_CLASS_NAMES:
        assert f"dice/{name}" in res


def test_reports_all_foreground_classes(num_classes):
    label = torch.randint(0, num_classes, (2, 1, 12, 12, 12))
    logits = torch.randn(2, num_classes, 12, 12, 12)
    m = SegMetrics(num_classes=num_classes)
    m.update(logits, label)
    res = m.aggregate()
    assert len([k for k in res if k.startswith("dice/")]) == len(FG_CLASS_NAMES)
    assert "dice_mean_fg" in res and "hd95_mean_fg" in res
