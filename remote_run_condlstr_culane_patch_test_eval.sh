#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
DATA_ROOT="${CONDLSTR_DATA_DIR:-/workspace/datasets}"
LOGS_NAME="${1:-culane_patch_enc24x42_1600x640_bs1}"
LOGS_DIR="${CONDLSTR_LOGS_DIR:-${REPO_ROOT}/logs/${LOGS_NAME}}"
TEST_DIR="${CONDLSTR_TEST_DIR:-${REPO_ROOT}/output/${LOGS_NAME}}"
NUM_WORKERS="${CONDLSTR_NUM_WORKERS:-2}"
BATCH_SIZE="${CONDLSTR_BATCH_SIZE:-1}"
IMG_W="${CONDLSTR_IMG_W:-1600}"
IMG_H="${CONDLSTR_IMG_H:-640}"
ENC_H="${CONDLSTR_ENC_H:-24}"
ENC_W="${CONDLSTR_ENC_W:-42}"
PRECISION="${CONDLSTR_PRECISION:-amp}"
SCORE_THRESH="${2:-0.7}"

cd "${REPO_ROOT}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate clrernet
fi

export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-max_split_size_mb:128}"

python -m py_compile \
  tools/test.py \
  modeling/inferences/lane/lane_det_2d.py \
  tools/metrics/lane/culane.py

python -u tools/test.py \
  -a CondLSTR2DRes34 \
  -d culane \
  -v v1.0 \
  -c 21 \
  -t lane_det_2d \
  --data-dir "${DATA_ROOT}" \
  --logs-dir "${LOGS_DIR}" \
  --checkpoint "${LOGS_DIR}/checkpoint.pth.tar" \
  --test-dir "${TEST_DIR}" \
  --split test \
  --img-size "${IMG_W}" "${IMG_H}" \
  --enc-src-hw "${ENC_H}" "${ENC_W}" \
  --score-thresh "${SCORE_THRESH}" \
  -b "${BATCH_SIZE}" \
  -j "${NUM_WORKERS}" \
  -p "${PRECISION}" \
  2>&1 | tee "test_${LOGS_NAME}_thr${SCORE_THRESH//./}.log"

python tools/metrics/lane/culane.py \
  --pred_dir "${TEST_DIR}" \
  --anno_dir "${DATA_ROOT}/culane" \
  --list "${DATA_ROOT}/culane/list/test.txt" \
  --width 30 \
  --official \
  --translate_v2 \
  --sequential \
  2>&1 | tee "metric_${LOGS_NAME}_thr${SCORE_THRESH//./}.log"
