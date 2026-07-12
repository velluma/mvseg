"""Transform pipeline tests (label integrity + shapes)."""

from __future__ import annotations

import numpy as np
import torch

from mvseg import NUM_CLASSES
from mvseg.data.transforms import eval_transforms, train_transforms


def _fake_meta_tensor_dict(spatial=(32, 32, 32)):
    """Emulate what LoadImaged would emit, so we can test the augment stages.

    We bypass file IO by constructing channel-first tensors directly and running
    only the post-load transforms via the public Compose (LoadImaged is skipped
    here; covered indirectly by the datamodule synthetic path)."""
    rng = np.random.default_rng(0)
    img = torch.from_numpy(rng.standard_normal((1, *spatial)).astype(np.float32))
    lab = torch.from_numpy(rng.integers(0, NUM_CLASSES, (1, *spatial)).astype(np.int64))
    return {"image": img, "label": lab}


def test_train_transforms_preserve_labels():
    from monai import transforms as T

    data = _fake_meta_tensor_dict()
    # Run only augmentation stages (normalize + spatial), not LoadImaged.
    pipeline = T.Compose(
        [
            T.NormalizeIntensityd(keys="image"),
            T.RandFlipd(keys=["image", "label"], prob=1.0, spatial_axis=0),
            T.RandAffined(
                keys=["image", "label"],
                prob=1.0,
                rotate_range=(0.1, 0.1, 0.1),
                mode=("bilinear", "nearest"),
            ),
        ]
    )
    out = pipeline(data)
    labels = torch.unique(out["label"]).tolist()
    # nearest interpolation must not introduce fractional / out-of-range classes
    assert all(float(v).is_integer() for v in labels)
    assert min(labels) >= 0 and max(labels) < NUM_CLASSES


def test_transform_builders_return_compose():
    assert train_transforms().__class__.__name__ == "Compose"
    assert eval_transforms().__class__.__name__ == "Compose"
