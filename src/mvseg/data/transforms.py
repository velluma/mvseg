"""MONAI transform pipelines for TEE volume/label pairs.

Keys used throughout: ``"image"`` (float intensity) and ``"label"`` (integer GT).
Label transforms always use nearest-neighbor interpolation to preserve class ids.
"""

from __future__ import annotations

from monai import transforms as T


def _normalize_transform(normalize: str) -> T.Transform:
    if normalize == "znorm":
        return T.NormalizeIntensityd(keys="image", nonzero=False, channel_wise=True)
    if normalize == "minmax":
        return T.ScaleIntensityd(keys="image", minv=0.0, maxv=1.0)
    raise ValueError(f"Unknown normalize mode: {normalize!r} (use 'znorm' or 'minmax')")


def _load_transforms() -> list[T.Transform]:
    """Load nrrd volume/label pair and standardize orientation/channel layout."""
    return [
        T.LoadImaged(keys=["image", "label"], reader="ITKReader", image_only=False),
        T.EnsureChannelFirstd(keys=["image", "label"]),
        T.Orientationd(keys=["image", "label"], axcodes="RAS"),
        T.EnsureTyped(keys=["image", "label"]),
    ]


def train_transforms(
    normalize: str = "znorm",
    augment: bool = True,
    patch_based: bool = False,
    spatial_size: tuple[int, int, int] = (128, 128, 128),
    num_samples: int = 2,
    num_classes: int = 5,
) -> T.Compose:
    tfms: list[T.Transform] = [*_load_transforms(), _normalize_transform(normalize)]

    if patch_based:
        tfms.append(
            T.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=spatial_size,
                pos=2,
                neg=1,
                num_samples=num_samples,
                allow_smaller=True,
            )
        )

    if augment:
        tfms += [
            T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            T.RandRotate90d(keys=["image", "label"], prob=0.3, max_k=3),
            T.RandAffined(
                keys=["image", "label"],
                prob=0.3,
                rotate_range=(0.26, 0.26, 0.26),  # ~15 deg
                scale_range=(0.1, 0.1, 0.1),
                mode=("bilinear", "nearest"),
                padding_mode="border",
            ),
            T.RandGaussianNoised(keys="image", prob=0.2, std=0.05),
            T.RandGaussianSmoothd(keys="image", prob=0.15),
            T.RandAdjustContrastd(keys="image", prob=0.2, gamma=(0.7, 1.5)),
        ]

    tfms.append(T.EnsureTyped(keys=["image", "label"]))
    return T.Compose(tfms)


def eval_transforms(normalize: str = "znorm") -> T.Compose:
    """Deterministic pipeline for validation/test/inference (no augmentation)."""
    return T.Compose([*_load_transforms(), _normalize_transform(normalize)])
