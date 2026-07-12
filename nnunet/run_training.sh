#!/usr/bin/env bash
# Train nnU-Net v2 on the MVSeg dataset (all 5 folds by default).
#
# Usage:  bash nnunet/run_training.sh <DATASET_ID> [CONFIG] [FOLDS...]
#   DATASET_ID : integer id used at conversion time (e.g. 1)
#   CONFIG     : 3d_fullres (default) | 3d_lowres | 2d | 3d_cascade_fullres
#   FOLDS      : space-separated fold indices (default: 0 1 2 3 4)
#
# Requires env vars: nnUNet_raw, nnUNet_preprocessed, nnUNet_results
set -euo pipefail

DATASET_ID="${1:?Provide dataset id, e.g. 1}"
CONFIG="${2:-3d_fullres}"
shift $(( $# > 1 ? 2 : 1 )) || true
FOLDS=("${@:-0 1 2 3 4}")

: "${nnUNet_raw:?Set nnUNet_raw}"
: "${nnUNet_preprocessed:?Set nnUNet_preprocessed}"
: "${nnUNet_results:?Set nnUNet_results}"

echo "Planning & preprocessing dataset ${DATASET_ID}..."
nnUNetv2_plan_and_preprocess -d "${DATASET_ID}" --verify_dataset_integrity

for FOLD in ${FOLDS[@]}; do
  echo "=== Training dataset ${DATASET_ID} | ${CONFIG} | fold ${FOLD} ==="
  nnUNetv2_train "${DATASET_ID}" "${CONFIG}" "${FOLD}"
done

echo "Done. Find the best configuration with:"
echo "  nnUNetv2_find_best_configuration ${DATASET_ID} -c ${CONFIG}"
