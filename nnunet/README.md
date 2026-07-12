# nnU-Net v2 Track

This directory wraps [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) for the MVSeg
dataset so it can be compared against the MONAI Residual UNet on identical data.

nnU-Net manages its own preprocessing, network configuration, and training, so we
only provide (1) a converter into nnU-Net raw format and (2) thin run wrappers.

## 0. Environment variables

nnU-Net locates data through three env vars. Set them once per shell (or in your
`docker-compose.yml`):

```bash
export nnUNet_raw=$PWD/nnUNet_raw
export nnUNet_preprocessed=$PWD/nnUNet_preprocessed
export nnUNet_results=$PWD/nnUNet_results
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"
```

## 1. Convert the local dataset

```bash
uv run python nnunet/convert_to_nnunet.py --data-dir data/raw --dataset-id 1
```

This creates `$nnUNet_raw/Dataset001_MVSeg/{imagesTr,labelsTr}` and a `dataset.json`
with our 5 labels (0=background, 1=anterior leaflet, 2=posterior leaflet,
3=mitral valve annulus, 4=aortic valve annulus) and a single `TEE` channel.

> Use `--symlink` to avoid duplicating large volumes on disk.

## 2. Plan, preprocess & train

```bash
# all in one (all 5 folds, 3d_fullres):
bash nnunet/run_training.sh 1 3d_fullres

# or manually:
uv run nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
uv run nnUNetv2_train 1 3d_fullres 0     # fold 0
```

nnU-Net v2 logs to `$nnUNet_results/.../training_log*.txt` and writes
`progress.png`. To also surface curves in **wandb**, wrap the command:

```bash
WANDB_PROJECT=mvseg wandb sync   # or parse progress.csv into a wandb run
```

(nnU-Net has no native wandb hook; the simplest reproducible option is to record
the final `summary.json` metrics — see step 4 — as a wandb run for comparison.)

## 3. Find best configuration & predict

```bash
uv run nnUNetv2_find_best_configuration 1 -c 3d_fullres
bash nnunet/run_inference.sh 1 3d_fullres <input_images_dir> preds_nnunet
```

`<input_images_dir>` must contain files named `<case>_0000.nrrd`. To predict on
our validation/test split images, first copy/symlink them with the `_0000` suffix
(the converter already does this for the training set under `imagesTr`).

## 4. Compare with the MONAI model

nnU-Net writes per-class Dice to `summary.json` inside each fold's validation
folder (`$nnUNet_results/Dataset001_MVSeg/.../validation/summary.json`). The class
indices there match our label map, so you can place them side-by-side with the
`test_metrics.csv` produced by `mvseg.evaluate`:

| structure | MONAI ResUNet (Dice) | nnU-Net (Dice) |
|-----------|----------------------|----------------|
| anterior leaflet | … | … |
| posterior leaflet | … | … |
| mitral valve annulus | … | … |
| aortic valve annulus | … | … |

Keep the **same held-out test cases** (`data/splits/splits.json`) out of nnU-Net
training folds to make the comparison fair.
