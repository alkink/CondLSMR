#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
DATA_ROOT="${CONDLSTR_DATA_DIR:-/workspace/datasets}"
LOGS_NAME="${2:-culane_patch_enc24x42_1600x640_bs1}"
LOGS_DIR="${CONDLSTR_LOGS_DIR:-${REPO_ROOT}/logs/${LOGS_NAME}}"
MAX_STEPS="${1:-150000}"
NUM_WORKERS="${CONDLSTR_NUM_WORKERS:-2}"
BATCH_SIZE="${CONDLSTR_BATCH_SIZE:-1}"
IMG_W="${CONDLSTR_IMG_W:-1600}"
IMG_H="${CONDLSTR_IMG_H:-640}"
ENC_H="${CONDLSTR_ENC_H:-24}"
ENC_W="${CONDLSTR_ENC_W:-42}"
PRECISION="${CONDLSTR_PRECISION:-amp}"
EXTRA_ARGS=()

if [[ "${3:-}" == "resume" ]]; then
  EXTRA_ARGS+=(--resume)
fi

cd "${REPO_ROOT}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate clrernet
fi

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-max_split_size_mb:128}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

python -m py_compile \
  tools/train.py \
  tools/test.py \
  engine/trainer.py \
  modeling/models/backbones/transformer/transformer.py \
  modeling/models/models/lane/cond_lstr_2d_res34.py \
  modeling/models/models/lane/cond_lstr_2d_res18.py \
  modeling/inferences/lane/lane_det_2d.py \
  tools/metrics/lane/culane.py

python -u tools/train.py \
  -a CondLSTR2DRes34 \
  -d culane \
  -v v1.0 \
  -c 21 \
  -t lane_det_2d \
  --data-dir "${DATA_ROOT}" \
  --logs-dir "${LOGS_DIR}" \
  --train-split train \
  --val-split val \
  --img-size "${IMG_W}" "${IMG_H}" \
  --enc-src-hw "${ENC_H}" "${ENC_W}" \
  -b "${BATCH_SIZE}" \
  -j "${NUM_WORKERS}" \
  -p "${PRECISION}" \
  -e 50 \
  --max-steps "${MAX_STEPS}" \
  --eval-epoch 5 \
  --save-steps 10000 \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "train_${LOGS_NAME}.log"
