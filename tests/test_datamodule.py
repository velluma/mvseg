"""DataModule tests using the synthetic path (no real data / files)."""

from __future__ import annotations

import torch

from mvseg import NUM_CLASSES
from mvseg.data.datamodule import MVSegDataModule
from mvseg.data.splits import extract_patient_id, make_splits


def test_synthetic_batches_have_expected_shape():
    dm = MVSegDataModule(
        synthetic=True,
        synthetic_num_train=4,
        synthetic_num_val=2,
        spatial_size=(32, 32, 32),
        batch_size=2,
        val_batch_size=1,
        num_workers=0,
        num_classes=NUM_CLASSES,
    )
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))
    assert batch["image"].shape == (2, 1, 32, 32, 32)
    assert batch["label"].shape == (2, 1, 32, 32, 32)
    assert batch["label"].dtype == torch.int64
    assert int(batch["label"].max()) < NUM_CLASSES

    val_batch = next(iter(dm.val_dataloader()))
    assert val_batch["image"].shape[0] == 1


# Realistic case ids: <patientID>_<imageID>_<frameNum>, ~3 frames x 2 images per patient.
def _synthetic_case_ids(n_patients: int = 30) -> list[str]:
    ids = []
    for p in range(n_patients):
        for img in range(2):
            for frame in range(3):
                ids.append(f"P{p:03d}_IMG{img}_{frame:02d}")
    return ids


def test_make_splits_is_deterministic_and_disjoint():
    ids = _synthetic_case_ids()
    s1 = make_splits(ids, seed=42)
    s2 = make_splits(ids, seed=42)
    assert s1.train == s2.train and s1.val == s2.val and s1.test == s2.test

    all_ids = set(s1.train) | set(s1.val) | set(s1.test)
    assert all_ids == set(ids)
    # disjoint at the case level
    assert not (set(s1.train) & set(s1.val))
    assert not (set(s1.train) & set(s1.test))
    assert not (set(s1.val) & set(s1.test))


def test_no_patient_leaks_across_splits():
    ids = _synthetic_case_ids()
    s = make_splits(ids, seed=7)

    def patients(cases):
        return {extract_patient_id(c) for c in cases}

    # a patient's frames must live entirely in one split
    assert not (patients(s.train) & patients(s.test))
    assert not (patients(s.train) & patients(s.val))
    assert not (patients(s.val) & patients(s.test))


def test_frozen_test_patients_are_pinned():
    ids = _synthetic_case_ids()
    frozen = ["P000", "P001", "P002"]
    s = make_splits(ids, seed=1, frozen_test_patients=frozen)
    assert {extract_patient_id(c) for c in s.test} == set(frozen)
    # frozen patients never appear in train/val
    assert not ({extract_patient_id(c) for c in s.train} & set(frozen))


def test_adding_patients_keeps_existing_assignment_stable():
    # Growing dataset: existing patients must not change split when new ones arrive.
    small = _synthetic_case_ids(20)
    big = _synthetic_case_ids(40)
    s_small = make_splits(small, seed=42)
    s_big = make_splits(big, seed=42)

    def split_of(splits, case):
        if case in splits.test:
            return "test"
        return "val" if case in splits.val else "train"

    for case in small:
        assert split_of(s_small, case) == split_of(s_big, case)


def test_make_splits_kfold():
    ids = _synthetic_case_ids()
    s = make_splits(ids, seed=0, n_folds=5)
    assert s.folds is not None and len(s.folds) == 5
