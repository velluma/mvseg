"""Sanity-check the dataset: shapes, spacing, label distribution, pairing.

Usage:
    python scripts/inspect_data.py --data-dir data/raw
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from mvseg import CLASS_NAMES
from mvseg.data.splits import extract_patient_id

try:
    import SimpleITK as sitk  # noqa: N813
except ImportError:  # pragma: no cover
    sitk = None


def _read(path: Path):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)  # (z, y, x)
    return arr, img.GetSpacing(), img.GetSize()


def main() -> None:
    p = argparse.ArgumentParser(description="Inspect the MVSeg dataset")
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--subdir", default="")
    p.add_argument("--image-suffix", default="_volume")
    p.add_argument("--label-suffix", default="_gt")
    p.add_argument("--file-ext", default=".nrrd")
    p.add_argument("--max-cases", type=int, default=0, help="0 = all")
    args = p.parse_args()

    if sitk is None:
        raise SystemExit("SimpleITK is required: uv sync --extra dev")

    base = Path(args.data_dir) / args.subdir if args.subdir else Path(args.data_dir)
    tail = f"{args.label_suffix}{args.file_ext}"
    case_ids = sorted(p.name[: -len(tail)] for p in base.glob(f"*{tail}"))
    if args.max_cases:
        case_ids = case_ids[: args.max_cases]
    if not case_ids:
        raise SystemExit(f"No '*{tail}' files under {base}")

    patients = {extract_patient_id(cid) for cid in case_ids}
    print(f"{len(patients)} patients, {len(case_ids)} labeled frames in {base}\n")
    shape_counter: Counter = Counter()
    global_labels: Counter = Counter()
    missing_labels = []
    problems = []

    for cid in case_ids:
        img_path = base / f"{cid}{args.image_suffix}{args.file_ext}"
        lab_path = base / f"{cid}{args.label_suffix}{args.file_ext}"
        if not img_path.exists():
            missing_labels.append(cid)
            continue
        img, spacing, size = _read(img_path)
        lab, _, lab_size = _read(lab_path)

        shape_counter[img.shape] += 1
        if img.shape != lab.shape:
            problems.append(f"{cid}: image {img.shape} != label {lab.shape}")

        uniq, counts = np.unique(lab, return_counts=True)
        for u, c in zip(uniq.tolist(), counts.tolist(), strict=False):
            global_labels[int(u)] += c
        unexpected = [u for u in uniq.tolist() if u not in range(len(CLASS_NAMES))]
        if unexpected:
            problems.append(f"{cid}: unexpected labels {unexpected}")

    print("Image shapes (z,y,x):")
    for shape, n in shape_counter.most_common():
        print(f"  {shape}: {n}")

    print("\nGlobal label voxel counts:")
    total = sum(global_labels.values()) or 1
    for idx, name in enumerate(CLASS_NAMES):
        cnt = global_labels.get(idx, 0)
        print(f"  {idx} {name:22s} {cnt:>14,d}  ({100 * cnt / total:6.3f}%)")

    if missing_labels:
        print(f"\n[WARN] {len(missing_labels)} labels without a volume: {missing_labels[:10]}")
    if problems:
        print(f"\n[WARN] {len(problems)} issues:")
        for msg in problems[:20]:
            print(f"  {msg}")
    if not missing_labels and not problems:
        print("\nAll checks passed.")


if __name__ == "__main__":
    main()
