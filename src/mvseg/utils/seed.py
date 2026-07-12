"""Reproducibility helpers."""

from __future__ import annotations

import os

import torch
from lightning.pytorch import seed_everything


def set_reproducibility(seed: int = 42, deterministic: bool = True) -> None:
    """Seed all RNGs and (optionally) enable deterministic algorithms.

    When ``deterministic`` is True, cuDNN determinism is enforced. Some 3D ops
    lack deterministic CUDA kernels; we set ``CUBLAS_WORKSPACE_CONFIG`` and use
    ``warn_only`` so training does not crash but non-determinism is surfaced.
    """
    seed_everything(seed, workers=True)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only: 3D transpose-conv / upsample kernels may be non-deterministic.
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True
