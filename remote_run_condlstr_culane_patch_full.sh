#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
cd "${REPO_ROOT}"

bash "${REPO_ROOT}/remote_run_condlstr_culane_patch_train.sh" "${1:-150000}" "${2:-culane_patch_enc24x42_1600x640_bs1}"
bash "${REPO_ROOT}/remote_run_condlstr_culane_patch_test_eval.sh" "${2:-culane_patch_enc24x42_1600x640_bs1}" "${3:-0.7}"
