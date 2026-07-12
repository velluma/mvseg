"""Shared fixtures: synthetic 128^3 volume/label pairs (no real data needed)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mvseg import NUM_CLASSES

SPATIAL = (32, 32, 32)  # small volumes keep CPU tests fast; pipeline is size-agnostic


@pytest.fixture
def spatial_size() -> tuple[int, int, int]:
    return SPATIAL


@pytest.fixture
def num_classes() -> int:
    return NUM_CLASSES


@pytest.fixture
def sample_batch(num_classes) -> dict:
    """A (B=2) batch of channel-first image/label tensors."""
    rng = np.random.default_rng(0)
    img = rng.standard_normal((2, 1, *SPATIAL)).astype(np.float32)
    lab = rng.integers(0, num_classes, size=(2, 1, *SPATIAL)).astype(np.int64)
    return {"image": torch.from_numpy(img), "label": torch.from_numpy(lab)}
