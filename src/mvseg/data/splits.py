"""Patient-level train/val/test split generation and loading.

Splitting is done at the **patient** level to prevent data leakage: a 4D image's
labeled frames — and all images belonging to the same patient — are highly
correlated, so every case of a patient goes entirely into one split.

Filenames follow::

    <patientID>_<imageID>_<frameNum>_volume.nrrd    # intensity volume
    <patientID>_<imageID>_<frameNum>_gt.nrrd         # ground-truth labels

The shared stem (``<patientID>_<imageID>_<frameNum>``) is the *case id*; its first
underscore-delimited token is the *patient id*.

Assignment is a deterministic function of the patient id (stable hashing), so when
new patients are added weekly, existing patients never change split. An optional
frozen list of test patients pins the held-out benchmark set.

Splits are stored as JSON and committed so every experiment (MONAI + nnU-Net) uses
the exact same partition.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


def extract_patient_id(case_id: str) -> str:
    """Patient id = first underscore-delimited token of the case id."""
    return case_id.split("_")[0]


def _patient_bucket(patient_id: str, seed: int) -> float:
    """Stable value in [0, 1) from a patient id — independent of dataset size."""
    h = hashlib.md5(f"{seed}:{patient_id}".encode()).hexdigest()
    return (int(h[:8], 16) % 10_000) / 10_000.0


def _patient_fold(patient_id: str, seed: int, n_folds: int) -> int:
    """Stable fold index for a patient id."""
    h = hashlib.md5(f"fold:{seed}:{patient_id}".encode()).hexdigest()
    return int(h[:8], 16) % n_folds


@dataclass
class Splits:
    train: list[str]
    val: list[str]
    test: list[str]
    # Optional k-fold assignment (over the non-test pool): fold index -> val case ids.
    folds: dict[str, list[str]] | None = None
    # Patient ids per split, for transparency / auditing.
    patients: dict[str, list[str]] | None = None

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict = {"train": self.train, "val": self.val, "test": self.test}
        if self.folds is not None:
            payload["folds"] = self.folds
        if self.patients is not None:
            payload["patients"] = self.patients
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> Splits:
        data = json.loads(Path(path).read_text())
        return cls(
            train=data["train"],
            val=data["val"],
            test=data.get("test", []),
            folds=data.get("folds"),
            patients=data.get("patients"),
        )


def list_case_ids(
    data_dir: str | Path,
    label_suffix: str = "_gt",
    file_ext: str = ".nrrd",
    subdir: str = "",
) -> list[str]:
    """Return sorted case ids (stems) for every labeled frame under ``data_dir``.

    A case is discovered from its label file ``<stem><label_suffix><file_ext>``.
    """
    base = Path(data_dir) / subdir if subdir else Path(data_dir)
    if not base.is_dir():
        raise FileNotFoundError(f"Data directory not found: {base}")
    tail = f"{label_suffix}{file_ext}"
    ids = sorted(p.name[: -len(tail)] for p in base.glob(f"*{tail}"))
    if not ids:
        raise FileNotFoundError(f"No '*{tail}' files under {base}")
    return ids


def group_by_patient(
    case_ids: list[str], patient_fn: Callable[[str], str] = extract_patient_id
) -> dict[str, list[str]]:
    """Map patient id -> sorted list of its case ids."""
    groups: dict[str, list[str]] = {}
    for cid in case_ids:
        groups.setdefault(patient_fn(cid), []).append(cid)
    return {p: sorted(cs) for p, cs in sorted(groups.items())}


def make_splits(
    case_ids: list[str],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    n_folds: int = 0,
    frozen_test_patients: list[str] | None = None,
    patient_fn: Callable[[str], str] = extract_patient_id,
) -> Splits:
    """Partition cases into train/val/test at the **patient** level.

    Assignment is a stable hash of the patient id, so adding patients later never
    reshuffles existing ones. If ``frozen_test_patients`` is given, those patients
    are the (pinned) test set and everyone else is hashed into train/val only;
    otherwise the test set is derived from the hash too.

    If ``n_folds > 1``, also computes k-fold val assignments over the non-test pool
    (patient-grouped) — aligning with nnU-Net's default 5-fold cross-validation.
    """
    groups = group_by_patient(case_ids, patient_fn)
    frozen = set(frozen_test_patients or [])
    have_frozen = len(frozen) > 0

    split_of: dict[str, str] = {}
    for patient in groups:
        if patient in frozen:
            split_of[patient] = "test"
            continue
        b = _patient_bucket(patient, seed)
        if have_frozen:  # test is pinned; split the rest into train/val
            split_of[patient] = "val" if b < val_frac else "train"
        elif b < test_frac:
            split_of[patient] = "test"
        elif b < test_frac + val_frac:
            split_of[patient] = "val"
        else:
            split_of[patient] = "train"

    cases: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    patients: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for patient, s in split_of.items():
        cases[s].extend(groups[patient])
        patients[s].append(patient)
    for d in (cases, patients):
        for k in d:
            d[k].sort()

    folds = None
    if n_folds and n_folds > 1:
        non_test = sorted(patients["train"] + patients["val"])
        folds = {str(k): [] for k in range(n_folds)}
        for patient in non_test:
            folds[str(_patient_fold(patient, seed, n_folds))].extend(groups[patient])
        for k in folds:
            folds[k].sort()

    return Splits(
        train=cases["train"], val=cases["val"], test=cases["test"], folds=folds, patients=patients
    )
