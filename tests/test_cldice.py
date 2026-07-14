"""Tests for the clDice topology loss and topology metrics.

Uses synthetic 3D "washer" rings: a closed ring (Betti-1 = 1) vs. rings with one
or two gaps, so we can check that the loss/metrics react to the topology change,
not just voxel overlap.
"""

from __future__ import annotations

import numpy as np
import torch

from mvseg.losses import (
    ClDiceAugmentedLoss,
    build_loss,
    soft_cldice_loss,
    soft_skeletonize,
)
from mvseg.metrics import _topology_stats


def make_ring(size: int = 32, r_in: float = 7.0, r_out: float = 10.0, z_thick: int = 3, gaps=()):
    """A ring in the XY plane extruded a few voxels in Z. ``gaps`` = angular
    (lo, hi) wedges (radians) removed to break the loop."""
    vol = np.zeros((size, size, size), dtype=np.float32)
    c = size / 2 - 0.5
    yy, xx = np.mgrid[0:size, 0:size]
    rr = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)
    ring = (rr >= r_in) & (rr <= r_out)
    ang = np.arctan2(yy - c, xx - c)
    for lo, hi in gaps:
        ring = ring & ~((ang >= lo) & (ang <= hi))
    zc = size // 2
    for z in range(zc - z_thick // 2, zc + z_thick // 2 + 1):
        vol[:, :, z] = ring
    return vol


def _as_5d(mask: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(mask.astype(np.float32))[None, None]


# --------------------------------------------------------------------------- #
# Soft skeleton / clDice loss
# --------------------------------------------------------------------------- #


def test_soft_skeleton_shape_and_nonneg():
    x = torch.rand(2, 1, 16, 16, 16)
    s = soft_skeletonize(x, iters=3)
    assert s.shape == x.shape
    assert float(s.min()) >= 0.0
    assert torch.isfinite(s).all()


def test_cldice_penalizes_broken_ring():
    intact = make_ring()
    broken = make_ring(gaps=[(-0.3, 0.3)])  # one wedge removed -> open ring
    target = _as_5d(intact)
    loss_intact = soft_cldice_loss(_as_5d(intact), target, iters=5)
    loss_broken = soft_cldice_loss(_as_5d(broken), target, iters=5)
    assert float(loss_intact) < 0.2  # near-perfect topology match
    assert float(loss_broken) > float(loss_intact)  # break is penalized


def test_cldice_loss_is_differentiable():
    logits = torch.randn(1, 5, 16, 16, 16, requires_grad=True)
    target = torch.randint(0, 5, (1, 1, 16, 16, 16))
    loss = build_loss("dice_ce_cldice", cldice_classes=[4], lambda_cldice=0.5, cldice_iters=3)(
        logits, target
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_cldice_zero_weight_matches_base():
    torch.manual_seed(0)
    logits = torch.randn(1, 5, 16, 16, 16)
    target = torch.randint(0, 5, (1, 1, 16, 16, 16))
    base = build_loss("dice_ce")
    aug = build_loss("dice_ce_cldice", cldice_classes=[4], lambda_cldice=0.0, cldice_iters=3)
    assert torch.allclose(base(logits, target), aug(logits, target), atol=1e-5)


def test_build_loss_returns_cldice_type():
    assert isinstance(build_loss("dice_ce_cldice"), ClDiceAugmentedLoss)
    assert isinstance(build_loss("dice_focal_cldice"), ClDiceAugmentedLoss)
    assert not isinstance(build_loss("dice_ce"), ClDiceAugmentedLoss)


# --------------------------------------------------------------------------- #
# Topology metrics
# --------------------------------------------------------------------------- #


def test_single_break_detected_by_loops_not_components():
    intact = make_ring().astype(bool)
    arc = make_ring(gaps=[(-0.3, 0.3)]).astype(bool)  # one gap: C-shape

    s_intact = _topology_stats(intact, intact)
    s_arc = _topology_stats(arc, intact)

    # A ring cut once is still ONE connected piece...
    assert s_intact["n_comp_pred"] == 1.0
    assert s_arc["n_comp_pred"] == 1.0
    # ...but the loop is gone -> loops metric catches it, betti0 does not.
    assert s_intact["n_loops_pred"] == 1.0
    assert s_arc["n_loops_pred"] == 0.0
    assert s_arc["loop_intact"] == 0.0
    assert s_arc["cldice"] < s_intact["cldice"]


def test_two_breaks_fragment_into_components():
    intact = make_ring().astype(bool)
    frag = make_ring(gaps=[(1.2, 1.9), (-1.9, -1.2)]).astype(bool)  # top+bottom gaps
    s = _topology_stats(frag, intact)
    assert s["n_comp_pred"] >= 2.0
    assert s["betti0_err"] >= 1.0
