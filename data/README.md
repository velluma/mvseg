# Data

The dataset lives **locally** and is not committed (see repo `.gitignore`). This
document is the contract every script/config assumes.

## Expected layout

```
data/
├── raw/
│   ├── P00123_IMG04_17_volume.nrrd   # 128×128×128, single-channel intensity
│   ├── P00123_IMG04_17_gt.nrrd       # 128×128×128, integer labels 0–4
│   ├── P00123_IMG04_23_volume.nrrd
│   ├── P00123_IMG04_23_gt.nrrd
│   └── ...
└── splits/
    └── splits.json                   # generated once, committed for reproducibility
```

Filename convention — a **case** is one labeled frame:

```
<patientID>_<imageID>_<frameNum>_volume.nrrd   # intensity
<patientID>_<imageID>_<frameNum>_gt.nrrd        # labels
└────────── case id (shared stem) ─────────┘
```

- **`patientID` = first underscore-delimited token.** Splitting is done at the
  **patient level** (all of a patient's images/frames stay in one split) to prevent
  data leakage — a 4D image's ~4 labeled frames are highly correlated.
- Labels are integers: `0=background, 1=anterior leaflet, 2=posterior leaflet,
  3=mitral valve annulus, 4=aortic valve annulus`.
- Volumes are `128³`; the pipeline does not resample by default.

> One 4D acquisition (`imageID`) has ~40 frames; only a handful (~4) are labeled,
> so only those `_volume`/`_gt` pairs need to exist on disk.

## Generate splits

```bash
uv run python scripts/prepare_splits.py --data-dir data/raw --out data/splits/splits.json
```

`splits/splits.json` **should be committed** so all experiments (MONAI + nnU-Net)
use the identical train/val/test partition. The raw volumes stay out of git.

### Growing dataset (weekly labels)

Patients are assigned by a **stable hash of the patient id**, so re-running
`prepare_splits.py` after new patients arrive never reshuffles existing ones.
To pin a benchmark test set, list its patient ids in a file and pass
`--frozen-test-file`:

```bash
uv run python scripts/prepare_splits.py --data-dir data/raw \
    --frozen-test-file data/splits/test_patients.txt --n-folds 5
```

Commit both `test_patients.txt` and the regenerated `splits.json`. Use `--n-folds 5`
to align with nnU-Net's default 5-fold cross-validation for fair comparison.

## Verify

```bash
uv run python scripts/inspect_data.py --data-dir data/raw
```

Reports shapes, spacing, per-class voxel counts, and flags mismatched or missing
pairs and unexpected label values.

## Future: DVC

To version the data later without changing code paths:

```bash
dvc init
dvc add data/raw
git add data/raw.dvc data/.gitignore
dvc remote add -d storage <s3://... | gs://... | /mnt/nas/...>
dvc push
```

The pipeline reads from `data/raw` regardless of whether it is a plain folder or a
DVC-tracked one, so no code changes are required.
