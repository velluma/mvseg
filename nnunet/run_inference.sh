#!/usr/bin/env bash
# Run nnU-Net v2 inference on a folder of TEE volumes.
#
# Usage:  bash nnunet/run_inference.sh <DATASET_ID> [CONFIG] <INPUT_DIR> <OUTPUT_DIR> [FOLDS]
#   INPUT_DIR must contain files named <case>_0000.nrrd (nnU-Net convention).
#
# Requires env vars: nnUNet_raw, nnUNet_preprocessed, nnUNet_results
set -euo pipefail

DATASET_ID="${1:?Provide dataset id}"
CONFIG="${2:-3d_fullres}"
INPUT_DIR="${3:?Provide input dir with *_0000.nrrd files}"
OUTPUT_DIR="${4:?Provide output dir}"
FOLDS="${5:-0 1 2 3 4}"

: "${nnUNet_results:?Set nnUNet_results}"

mkdir -p "${OUTPUT_DIR}"

echo "Predicting ${INPUT_DIR} -> ${OUTPUT_DIR} (dataset ${DATASET_ID}, ${CONFIG})"
nnUNetv2_predict \
  -i "${INPUT_DIR}" \
  -o "${OUTPUT_DIR}" \
  -d "${DATASET_ID}" \
  -c "${CONFIG}" \
  -f ${FOLDS}

echo "Predictions written to ${OUTPUT_DIR}"
