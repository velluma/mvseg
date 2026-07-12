"""LightningDataModule for the TEE mitral valve dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from monai.data import CacheDataset, DataLoader, Dataset, list_data_collate

from mvseg.data.splits import Splits, list_case_ids, make_splits
from mvseg.data.transforms import eval_transforms, train_transforms


class _SyntheticDataset(torch.utils.data.Dataset):
    """Random 128^3 volumes + label maps — for smoke tests / CI (no files needed)."""

    def __init__(self, n: int, spatial_size, num_classes: int, seed: int = 0):
        self.n = n
        self.spatial_size = tuple(spatial_size)
        self.num_classes = num_classes
        self.seed = seed

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        rng = np.random.default_rng(self.seed + idx)
        img = rng.standard_normal((1, *self.spatial_size)).astype(np.float32)
        # A few random foreground blobs so Dice is well-defined.
        lab = np.zeros((1, *self.spatial_size), dtype=np.int64)
        for cls in range(1, self.num_classes):
            c = rng.integers(20, 108, size=3)
            r = int(rng.integers(6, 14))
            zz, yy, xx = np.ogrid[
                : self.spatial_size[0], : self.spatial_size[1], : self.spatial_size[2]
            ]
            sphere = (zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2 <= r**2
            lab[0][sphere] = cls
        return {
            "image": torch.from_numpy(img),
            "label": torch.from_numpy(lab),
            "case_id": f"synthetic_{idx:03d}",
        }


class MVSegDataModule(LightningDataModule):
    """Loads paired image/label nrrd volumes according to a committed split file."""

    def __init__(
        self,
        data_dir: str = "data/raw",
        splits_file: str = "data/splits/splits.json",
        subdir: str = "",
        image_suffix: str = "_volume",
        label_suffix: str = "_gt",
        file_ext: str = ".nrrd",
        batch_size: int = 2,
        val_batch_size: int = 1,
        num_workers: int = 4,
        pin_memory: bool = True,
        spatial_size: tuple[int, int, int] = (128, 128, 128),
        patch_based: bool = False,
        num_samples: int = 2,
        normalize: str = "znorm",
        cache_rate: float = 0.0,
        augment: bool = True,
        num_classes: int = 5,
        seed: int = 42,
        synthetic: bool = False,
        synthetic_num_train: int = 4,
        synthetic_num_val: int = 2,
    ):
        super().__init__()
        self.save_hyperparameters()
        self._splits: Splits | None = None

    # ------------------------------------------------------------------ utils
    def _base_dir(self) -> Path:
        h = self.hparams
        return Path(h.data_dir) / h.subdir if h.subdir else Path(h.data_dir)

    def _records(self, case_ids: list[str]) -> list[dict]:
        h = self.hparams
        base = self._base_dir()
        return [
            {
                "image": str(base / f"{cid}{h.image_suffix}{h.file_ext}"),
                "label": str(base / f"{cid}{h.label_suffix}{h.file_ext}"),
                "case_id": cid,
            }
            for cid in case_ids
        ]

    def _load_or_make_splits(self) -> Splits:
        h = self.hparams
        splits_path = Path(h.splits_file)
        if splits_path.is_file():
            return Splits.from_json(splits_path)
        # Fall back to an on-the-fly deterministic patient-level split.
        case_ids = list_case_ids(h.data_dir, h.label_suffix, h.file_ext, h.subdir)
        return make_splits(case_ids, seed=h.seed)

    # ----------------------------------------------------------- lightning API
    def setup(self, stage: str | None = None) -> None:
        h = self.hparams
        if h.synthetic:
            return  # datasets built lazily in the *_dataloader methods
        self._splits = self._load_or_make_splits()

    def _build_dataset(self, records: list[dict], train: bool):
        h = self.hparams
        tfm = (
            train_transforms(
                normalize=h.normalize,
                augment=h.augment,
                patch_based=h.patch_based,
                spatial_size=tuple(h.spatial_size),
                num_samples=h.num_samples,
                num_classes=h.num_classes,
            )
            if train
            else eval_transforms(normalize=h.normalize)
        )
        if h.cache_rate and h.cache_rate > 0:
            return CacheDataset(
                data=records, transform=tfm, cache_rate=h.cache_rate, num_workers=h.num_workers
            )
        return Dataset(data=records, transform=tfm)

    # ------------------------------------------------------------- dataloaders
    def train_dataloader(self) -> DataLoader:
        h = self.hparams
        if h.synthetic:
            ds = _SyntheticDataset(h.synthetic_num_train, h.spatial_size, h.num_classes, seed=1)
        else:
            ds = self._build_dataset(self._records(self._splits.train), train=True)
        return DataLoader(
            ds,
            batch_size=h.batch_size,
            shuffle=True,
            num_workers=h.num_workers,
            pin_memory=h.pin_memory,
            collate_fn=list_data_collate,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        h = self.hparams
        if h.synthetic:
            ds = _SyntheticDataset(h.synthetic_num_val, h.spatial_size, h.num_classes, seed=100)
        else:
            ds = self._build_dataset(self._records(self._splits.val), train=False)
        return DataLoader(
            ds,
            batch_size=h.val_batch_size,
            shuffle=False,
            num_workers=h.num_workers,
            pin_memory=h.pin_memory,
            collate_fn=list_data_collate,
        )

    def test_dataloader(self) -> DataLoader:
        h = self.hparams
        if h.synthetic:
            ds = _SyntheticDataset(h.synthetic_num_val, h.spatial_size, h.num_classes, seed=200)
        else:
            records = self._records(self._splits.test or self._splits.val)
            ds = self._build_dataset(records, train=False)
        return DataLoader(
            ds,
            batch_size=h.val_batch_size,
            shuffle=False,
            num_workers=h.num_workers,
            pin_memory=h.pin_memory,
            collate_fn=list_data_collate,
        )
