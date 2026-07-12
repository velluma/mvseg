"""Convert the local nrrd dataset into nnU-Net v2 raw format.

Produces:
    $nnUNet_raw/DatasetXXX_MVSeg/
        imagesTr/<case>_0000.nrrd
        labelsTr/<case>.nrrd
        dataset.json

Usage:
    export nnUNet_raw=$PWD/nnUNet_raw
    python nnunet/convert_to_nnunet.py --data-dir data/raw --dataset-id 1

Notes:
- nnU-Net expects a single modality suffix ``_0000`` on images.
- Labels must be consecutive integers starting at 0 (background) — our GT already is.
- We copy files (no resampling); nnU-Net handles preprocessing itself.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

# Must match src/mvseg/__init__.py CLASS_NAMES ordering.
LABELS = {
    "background": 0,
    "anterior_leaflet": 1,
    "posterior_leaflet": 2,
    "mitral_valve_annulus": 3,
    "aortic_valve_annulus": 4,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert nrrd dataset to nnU-Net v2 raw format")
    p.add_argument("--data-dir", default="data/raw", help="dir with *_volume/*_gt nrrd files")
    p.add_argument("--subdir", default="")
    p.add_argument("--image-suffix", default="_volume")
    p.add_argument("--label-suffix", default="_gt")
    p.add_argument("--file-ext", default=".nrrd")
    p.add_argument("--dataset-id", type=int, default=1)
    p.add_argument("--dataset-name", default="MVSeg")
    p.add_argument(
        "--nnunet-raw",
        default=os.environ.get("nnUNet_raw"),  # noqa: SIM112 - nnU-Net's canonical env var name
        help="nnUNet_raw root (defaults to $nnUNet_raw)",
    )
    p.add_argument("--symlink", action="store_true", help="symlink instead of copy")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.nnunet_raw:
        raise SystemExit("Set $nnUNet_raw or pass --nnunet-raw")

    base = Path(args.data_dir) / args.subdir if args.subdir else Path(args.data_dir)
    if not base.is_dir():
        raise SystemExit(f"Expected {base} to exist")

    dataset_folder = f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    out_root = Path(args.nnunet_raw) / dataset_folder
    out_img = out_root / "imagesTr"
    out_lab = out_root / "labelsTr"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lab.mkdir(parents=True, exist_ok=True)

    tail = f"{args.label_suffix}{args.file_ext}"
    case_ids = sorted(p.name[: -len(tail)] for p in base.glob(f"*{tail}"))
    if not case_ids:
        raise SystemExit(f"No '*{tail}' files under {base}")

    def _place(src: Path, dst: Path) -> None:
        if dst.exists():
            dst.unlink()
        if args.symlink:
            dst.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dst)

    n = 0
    for cid in case_ids:
        img_src = base / f"{cid}{args.image_suffix}{args.file_ext}"
        lab_src = base / f"{cid}{args.label_suffix}{args.file_ext}"
        if not img_src.exists():
            print(f"[skip] missing volume for {cid}")
            continue
        _place(img_src, out_img / f"{cid}_0000{args.file_ext}")
        _place(lab_src, out_lab / f"{cid}{args.file_ext}")
        n += 1

    dataset_json = {
        "channel_names": {"0": "TEE"},
        "labels": LABELS,
        "numTraining": n,
        "file_ending": args.file_ext,
        "overwrite_image_reader_writer": "SimpleITKIO",
    }
    (out_root / "dataset.json").write_text(json.dumps(dataset_json, indent=2))

    print(f"Wrote {n} cases to {out_root}")
    print("Next:")
    print(f"  nnUNetv2_plan_and_preprocess -d {args.dataset_id} --verify_dataset_integrity")


if __name__ == "__main__":
    main()
