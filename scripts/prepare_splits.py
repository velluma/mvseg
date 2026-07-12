"""Generate a version-pinned, patient-level train/val/test split file.

Splitting is at the patient level (see mvseg.data.splits) to avoid leaking a
patient's frames/images across splits. Assignment is a stable hash of the patient
id, so re-running after new patients are added keeps existing patients in place.

Usage:
    python scripts/prepare_splits.py --data-dir data/raw --out data/splits/splits.json
    python scripts/prepare_splits.py --data-dir data/raw --n-folds 5
    python scripts/prepare_splits.py --data-dir data/raw --frozen-test-file test_patients.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mvseg.data.splits import extract_patient_id, group_by_patient, list_case_ids, make_splits


def _read_frozen(path: str | None) -> list[str] | None:
    if not path:
        return None
    text = Path(path).read_text()
    # accept newline- or comma-separated patient ids
    return sorted({tok.strip() for tok in text.replace(",", "\n").splitlines() if tok.strip()})


def main() -> None:
    p = argparse.ArgumentParser(description="Create patient-level train/val/test splits")
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--subdir", default="")
    p.add_argument("--label-suffix", default="_gt")
    p.add_argument("--file-ext", default=".nrrd")
    p.add_argument("--out", default="data/splits/splits.json")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--n-folds", type=int, default=0, help="0 disables k-fold assignments")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--frozen-test-file",
        default=None,
        help="file with pinned test patient ids (newline/comma separated)",
    )
    args = p.parse_args()

    case_ids = list_case_ids(args.data_dir, args.label_suffix, args.file_ext, args.subdir)
    groups = group_by_patient(case_ids, extract_patient_id)
    splits = make_splits(
        case_ids,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        n_folds=args.n_folds,
        frozen_test_patients=_read_frozen(args.frozen_test_file),
    )
    splits.to_json(args.out)

    pats = splits.patients or {}
    print(f"{len(groups)} patients, {len(case_ids)} labeled frames")
    for name in ("train", "val", "test"):
        n_pat = len(pats.get(name, []))
        n_case = len(getattr(splits, name))
        print(f"  {name:5s}  patients={n_pat:4d}  frames={n_case:5d}")
    if splits.folds:
        print(f"  {len(splits.folds)}-fold assignments (patient-grouped) written")
    print(f"Wrote {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
