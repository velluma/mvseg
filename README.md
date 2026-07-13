# MVSeg — 3D TEE Mitral Valve Segmentation

Multi-class segmentation of the mitral valve apparatus from 3D transesophageal
echocardiography (TEE) volumes.

- **Input**: `128 × 128 × 128` single-channel intensity volumes (`.nrrd`)
- **Ground truth**: same-size label volume with 5 classes:

  | label | structure |
  |-------|-----------|
  | 0 | background |
  | 1 | anterior leaflet |
  | 2 | posterior leaflet |
  | 3 | mitral valve annulus |
  | 4 | aortic valve annulus |

Two experiment tracks share the same data and evaluation protocol:

1. **MONAI + PyTorch Lightning + wandb** — a 3D Residual UNet, fully configured
   with Hydra (`src/mvseg/`).
2. **nnU-Net v2** — wrapper scripts + a documented workflow (`nnunet/`).

Designed to be **reproducible in a medtech setting**: pinned dependencies (`uv`),
a CUDA Docker image, fixed data splits, deterministic seeding, and CI.

> ⚠️ Research use only — not a medical device.

---

## 1. Setup

We use [`uv`](https://docs.astral.sh/uv/) for reproducible environments.

```bash
# install uv (see https://docs.astral.sh/uv/getting-started/installation/)
# then:
uv sync --extra dev          # creates .venv from uv.lock
uv run python -c "import monai, lightning, wandb, hydra; print('ok')"
```

### GPU / CUDA wheels

`torch` is **pinned to a CUDA 11.8 build** (`torch==2.4.1+cu118`, via
`[tool.uv.sources]` in `pyproject.toml`) so it runs on **Tesla P100** and other
Pascal (sm_60) GPUs. Newer torch/CUDA wheels drop Pascal kernels and would fail
with `no kernel image is available for execution on the device`. The cu118 build
only needs driver `>= 450`, so it also works on older servers. Check yours:

```bash
nvidia-smi        # "Driver Version" must be >= 450 (cu118) / >= 525 (cu121)
```

To target a newer GPU with CUDA 12.1 instead, edit the `pytorch-cu118` index URL
in `pyproject.toml` to `.../whl/cu121`, bump the `torch` pin, change the Dockerfile
base image to `nvidia/cuda:12.1.1-...`, and re-run `uv lock`.

### Docker (recommended for training)

```bash
docker compose build
MVSEG_DATA_DIR=/abs/path/to/data docker compose run --rm mvseg bash
# inside the container:
uv run python -m mvseg.train experiment=resunet_baseline
```

---

## 2. Data layout

The dataset is assumed to live locally (DVC can be layered on later — see
[`data/README.md`](data/README.md)). Expected layout:

```
data/
├── raw/
│   ├── <patientID>_<imageID>_<frameNum>_volume.nrrd   # intensity
│   ├── <patientID>_<imageID>_<frameNum>_gt.nrrd        # labels 0–4
│   └── ...
└── splits/
    └── splits.json                                     # generated, version-pinned
```

Each labeled frame is one `_volume`/`_gt` pair sharing a stem. Splitting is
**patient-level** (first `_`-token = patient id) to prevent leakage — see
[`data/README.md`](data/README.md). Generate splits once and commit `splits.json`:

```bash
uv run python scripts/prepare_splits.py --data-dir data/raw --out data/splits/splits.json
uv run python scripts/inspect_data.py   --data-dir data/raw   # sanity report
```

---

## 3. Train / evaluate / predict (MONAI + Lightning)

Configuration is managed by **Hydra** (`configs/`). Override any value on the CLI.

```bash
# baseline Residual UNet
uv run python -m mvseg.train experiment=resunet_baseline

# override on the fly
uv run python -m mvseg.train model.channels=[32,64,128,256,512] trainer.max_epochs=500

# quick smoke test (no real data needed if data.synthetic=true)
uv run python -m mvseg.train trainer.fast_dev_run=true data.synthetic=true

# evaluate a checkpoint on the test split
uv run python -m mvseg.evaluate ckpt_path=outputs/<run>/checkpoints/best.ckpt

# predict on a single volume or a folder -> writes .nrrd label maps
uv run python -m mvseg.predict ckpt_path=... input=data/raw/P00123_IMG04_17_volume.nrrd output=preds/
```

Metrics logged to **wandb**: total & per-class Dice (anterior/posterior leaflet,
MV/AV annulus), 95% Hausdorff distance, plus mid-slice prediction overlays.

Configure wandb once:

```bash
export WANDB_API_KEY=...        # or `wandb login`
export WANDB_PROJECT=mvseg
# disable during dev:  uv run python -m mvseg.train logger=none
```

---

## 4. nnU-Net v2 track

See [`nnunet/README.md`](nnunet/README.md) for the full workflow. In short:

```bash
export nnUNet_raw=$PWD/nnUNet_raw
export nnUNet_preprocessed=$PWD/nnUNet_preprocessed
export nnUNet_results=$PWD/nnUNet_results

uv run python nnunet/convert_to_nnunet.py --data-dir data/raw --dataset-id 1
uv run nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
bash nnunet/run_training.sh 1 3d_fullres
bash nnunet/run_inference.sh 1 3d_fullres $nnUNet_raw/Dataset001_MVSeg/imagesTr preds_nnunet
```

---

## 5. Reproducibility checklist

- [x] Pinned dependencies via `uv.lock`
- [x] CUDA Docker image (`Dockerfile`, `docker-compose.yml`)
- [x] Deterministic seeding (`configs/config.yaml: seed`, `deterministic`)
- [x] Fixed, version-pinned data splits (`data/splits/splits.json`)
- [x] Every run snapshots its resolved config (Hydra `outputs/`) + logs config/code to wandb
- [x] CI runs lint + CPU smoke tests on every push (`.github/workflows/ci.yml`)

---

## 6. Development

```bash
make lint     # ruff check + format check
make fmt      # auto-format
make test     # pytest (CPU smoke tests, synthetic data — no GPU/real data needed)
pre-commit install
```

## Project layout

```
src/mvseg/     Lightning + MONAI training package
configs/       Hydra config hierarchy
nnunet/        nnU-Net v2 conversion + run wrappers
scripts/       data inspection & split generation
tests/         CPU smoke tests on synthetic 128³ volumes
```
