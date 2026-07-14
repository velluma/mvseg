"""Segmentation metrics: per-class Dice, 95% Hausdorff distance, and (optionally)
topology metrics for ring-/tube-shaped classes.

We report foreground classes individually (anterior/posterior leaflet, MV/AV
annulus) plus a foreground mean, since a single averaged score hides failures on
the small, clinically important structures.

Topology metrics (enabled per-class via ``topo_classes``) quantify whether a
thin structure stays *connected* — the thing Dice/HD95 are blind to and clDice
training targets:

* ``topo/{cls}/betti0_err`` — |#components(pred) - #components(GT)| (26-conn).
  0 means the prediction has the same number of connected pieces as the GT.
  Catches *fragmentation* (ring split into >= 2 pieces) and spurious islands.
* ``topo/{cls}/n_comp_pred`` — mean number of predicted components (ideal: 1 for
  a single ring).
* ``topo/{cls}/connected_rate`` — fraction of cases where the component count
  matches the GT.
* ``topo/{cls}/n_loops_pred`` — Betti-1 (number of loops/holes) of the
  prediction, via the Euler characteristic assuming no enclosed cavities. A
  closed ring has 1 loop; a ring cut *once* stays one connected piece but its
  loop count drops to 0 — so this (not betti0) is what detects a single break.
* ``topo/{cls}/loop_intact_rate`` — fraction of cases whose loop count matches
  the GT (**the primary "ring didn't break" metric**).
* ``cldice/{cls}`` — hard centerline-Dice (skeleton-based overlap), higher
  better; sensitive to breaks of any severity.
"""

from __future__ import annotations

import numpy as np
import torch
from monai.data import decollate_batch
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete, Compose

from mvseg import CLASS_NAMES, FG_CLASS_NAMES


class SegMetrics:
    """Accumulates Dice/HD95 (and optional topology metrics) over an epoch.

    Usage:
        m = SegMetrics(num_classes=5, topo_classes=[4])
        m.update(logits, labels)     # per step
        results = m.aggregate()      # {"dice_mean_fg": ..., "topo/aortic_valve_annulus/...": ...}
        m.reset()
    """

    def __init__(
        self,
        num_classes: int = 5,
        compute_hd95: bool = True,
        topo_classes: list[int] | None = None,
    ):
        self.num_classes = num_classes
        self.compute_hd95 = compute_hd95
        self.topo_classes = [int(c) for c in (topo_classes or [])]
        self.dice = DiceMetric(include_background=False, reduction="mean_batch")
        self.hd95 = (
            HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean_batch")
            if compute_hd95
            else None
        )
        self._post_pred = Compose([AsDiscrete(argmax=True, to_onehot=num_classes)])
        self._post_label = Compose([AsDiscrete(to_onehot=num_classes)])
        self._topo_keys = (
            "betti0_err",
            "n_comp_pred",
            "connected",
            "n_loops_pred",
            "loop_intact",
            "cldice",
        )
        self._topo: dict[int, dict[str, list[float]]] = {
            c: {k: [] for k in self._topo_keys} for c in self.topo_classes
        }

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        preds = [self._post_pred(p) for p in decollate_batch(logits)]
        gts = [self._post_label(g) for g in decollate_batch(labels)]
        self.dice(y_pred=preds, y=gts)
        if self.hd95 is not None:
            self.hd95(y_pred=preds, y=gts)
        if self.topo_classes:
            self._update_topology(logits, labels)

    def _update_topology(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        # Hard argmax prediction and integer GT, per sample, on CPU numpy.
        pred_lbl = torch.argmax(logits, dim=1).detach().cpu().numpy()  # (B, H, W, D)
        gt = labels.detach().cpu().numpy()
        gt = gt[:, 0] if gt.ndim == pred_lbl.ndim + 1 else gt  # squeeze channel
        for b in range(pred_lbl.shape[0]):
            for c in self.topo_classes:
                gt_mask = gt[b] == c
                if not gt_mask.any():
                    continue  # class absent in this (patch) sample -> skip
                pred_mask = pred_lbl[b] == c
                stats = _topology_stats(pred_mask, gt_mask)
                for k, v in stats.items():
                    self._topo[c][k].append(v)

    def aggregate(self) -> dict[str, float]:
        out: dict[str, float] = {}
        dice_vals = _to_list(self.dice.aggregate())
        for name, val in zip(FG_CLASS_NAMES, dice_vals, strict=False):
            out[f"dice/{name}"] = val
        out["dice_mean_fg"] = float(sum(dice_vals) / len(dice_vals)) if dice_vals else 0.0

        if self.hd95 is not None:
            hd_vals = _to_list(self.hd95.aggregate())
            for name, val in zip(FG_CLASS_NAMES, hd_vals, strict=False):
                out[f"hd95/{name}"] = val
            finite = [v for v in hd_vals if v == v and v != float("inf")]
            out["hd95_mean_fg"] = float(sum(finite) / len(finite)) if finite else float("nan")

        for c in self.topo_classes:
            name = CLASS_NAMES[c]
            acc = self._topo[c]
            if acc["betti0_err"]:
                out[f"topo/{name}/betti0_err"] = _mean(acc["betti0_err"])
                out[f"topo/{name}/n_comp_pred"] = _mean(acc["n_comp_pred"])
                out[f"topo/{name}/connected_rate"] = _mean(acc["connected"])
                out[f"topo/{name}/n_loops_pred"] = _mean(acc["n_loops_pred"])
                out[f"topo/{name}/loop_intact_rate"] = _mean(acc["loop_intact"])
                out[f"cldice/{name}"] = _mean(acc["cldice"])
        return out

    def reset(self) -> None:
        self.dice.reset()
        if self.hd95 is not None:
            self.hd95.reset()
        for acc in self._topo.values():
            for lst in acc.values():
                lst.clear()


# --------------------------------------------------------------------------- #
# Topology helpers
# --------------------------------------------------------------------------- #


def _topology_stats(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict[str, float]:
    """Betti-0 (components) + Betti-1 (loops) agreement + hard clDice for a mask."""
    from scipy import ndimage
    from skimage.measure import euler_number

    conn = np.ones((3, 3, 3), dtype=int)  # 26-connectivity in 3D
    n_pred = int(ndimage.label(pred_mask, structure=conn)[1])
    n_gt = int(ndimage.label(gt_mask, structure=conn)[1])
    # Betti-1 via Euler characteristic: chi = b0 - b1 + b2. For a thin annulus we
    # assume no enclosed cavities (b2 = 0), so b1 = b0 - chi.
    loops_pred = n_pred - int(euler_number(pred_mask, connectivity=3))
    loops_gt = n_gt - int(euler_number(gt_mask, connectivity=3))
    return {
        "betti0_err": float(abs(n_pred - n_gt)),
        "n_comp_pred": float(n_pred),
        "connected": float(n_pred == n_gt),
        "n_loops_pred": float(loops_pred),
        "loop_intact": float(loops_pred == loops_gt),
        "cldice": _hard_cldice(pred_mask, gt_mask),
    }


def _hard_cldice(pred_mask: np.ndarray, gt_mask: np.ndarray, smooth: float = 1e-6) -> float:
    """Deterministic clDice using morphological skeletons (evaluation-time)."""
    from skimage.morphology import skeletonize

    if not pred_mask.any():
        return 0.0
    skel_pred = skeletonize(pred_mask)
    skel_gt = skeletonize(gt_mask)
    tprec = (float((skel_pred & gt_mask).sum()) + smooth) / (float(skel_pred.sum()) + smooth)
    tsens = (float((skel_gt & pred_mask).sum()) + smooth) / (float(skel_gt.sum()) + smooth)
    denom = tprec + tsens
    return float(2.0 * tprec * tsens / denom) if denom > 0 else 0.0


def _mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def _to_list(t) -> list[float]:
    if isinstance(t, torch.Tensor):
        return [float(x) for x in t.flatten().tolist()]
    return [float(t)]
