#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
INCLUDE_OPENCV=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: bash clean_artifacts.sh [--dry-run] [--include-opencv]

Removes only generated local artifacts under the CondLSTR repo.

Default targets:
  - logs/
  - output/
  - build/
  - temp/
  - all __pycache__/ directories
  - *.pyc / *.pyo
  - root-level *.log
  - core / core.* / *.core

Optional:
  --include-opencv   also remove ignored opencv/ directory
  --dry-run          print what would be removed without deleting
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --include-opencv)
      INCLUDE_OPENCV=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

cd "${REPO_ROOT}"

TARGETS=()
for path in logs output build temp; do
  if [[ -e "${path}" ]]; then
    TARGETS+=("${path}")
  fi
done

while IFS= read -r path; do
  TARGETS+=("${path}")
done < <(find . -type d -name '__pycache__' -print)

while IFS= read -r path; do
  TARGETS+=("${path}")
done < <(find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -print)

while IFS= read -r path; do
  TARGETS+=("${path}")
done < <(find . -maxdepth 1 -type f \( -name '*.log' -o -name 'core' -o -name 'core.*' -o -name '*.core' \) -print)

if [[ "${INCLUDE_OPENCV}" -eq 1 && -e opencv ]]; then
  TARGETS+=("opencv")
fi

if [[ "${#TARGETS[@]}" -eq 0 ]]; then
  echo "No generated artifacts found."
  exit 0
fi

echo "Artifacts selected for removal:"
printf '  %s\n' "${TARGETS[@]}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "Dry run only. Nothing removed."
  exit 0
fi

rm -rf -- "${TARGETS[@]}"
echo "Cleanup complete."
