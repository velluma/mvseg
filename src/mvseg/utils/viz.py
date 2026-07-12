"""Segmentation visualization for wandb logging."""

from __future__ import annotations

import numpy as np
import torch

# Distinct RGB colors per class (background transparent). Order == label index.
_CLASS_COLORS = np.array(
    [
        [0, 0, 0],  # 0 background
        [230, 25, 75],  # 1 anterior leaflet   (red)
        [60, 180, 75],  # 2 posterior leaflet  (green)
        [0, 130, 200],  # 3 mitral annulus     (blue)
        [245, 130, 48],  # 4 aortic annulus     (orange)
    ],
    dtype=np.uint8,
)


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().float().numpy()


def _mid_slice(vol: np.ndarray, axis: int = 2) -> np.ndarray:
    idx = vol.shape[axis] // 2
    return np.take(vol, idx, axis=axis)


def colorize_label(label_slice: np.ndarray) -> np.ndarray:
    """Map a 2D integer label slice to an RGB image."""
    label_slice = label_slice.astype(int)
    label_slice = np.clip(label_slice, 0, len(_CLASS_COLORS) - 1)
    return _CLASS_COLORS[label_slice]


def overlay_prediction(
    image: torch.Tensor,
    label: torch.Tensor,
    pred: torch.Tensor,
    axis: int = 2,
    alpha: float = 0.5,
) -> np.ndarray:
    """Build a side-by-side [image | GT overlay | prediction overlay] RGB panel.

    Args:
        image: (H, W, D) or (1, H, W, D) intensity volume.
        label: (H, W, D) integer GT volume.
        pred:  (H, W, D) integer predicted volume.

    Returns:
        (H, 3*W, 3) uint8 array suitable for ``wandb.Image``.
    """
    img = _to_numpy(image).squeeze()
    lab = _to_numpy(label).squeeze()
    prd = _to_numpy(pred).squeeze()

    img2d = _mid_slice(img, axis)
    lab2d = _mid_slice(lab, axis)
    prd2d = _mid_slice(prd, axis)

    # Normalize intensity slice to 0-255 grayscale RGB.
    lo, hi = float(img2d.min()), float(img2d.max())
    denom = (hi - lo) if hi > lo else 1.0
    gray = ((img2d - lo) / denom * 255.0).astype(np.uint8)
    gray_rgb = np.stack([gray] * 3, axis=-1)

    def _blend(mask2d: np.ndarray) -> np.ndarray:
        color = colorize_label(mask2d)
        fg = (mask2d > 0)[..., None]
        blended = gray_rgb.astype(np.float32)
        blended = np.where(fg, (1 - alpha) * blended + alpha * color.astype(np.float32), blended)
        return blended.astype(np.uint8)

    panel = np.concatenate([gray_rgb, _blend(lab2d), _blend(prd2d)], axis=1)
    return panel
