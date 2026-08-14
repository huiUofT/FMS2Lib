#!/bin/bash
# Train gen+inten for a prepared dataset (docx hyper-params).
# Example:
#   export ICEBERG_PYTHON="$(which python)"
#   bash sh/foriceberg/train_from_dataset.sh --dataset-name F_ICEBERG_train100 --gpu
set -euo pipefail
GITHUB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FORICEBERG_SCRIPT="${GITHUB_ROOT}/script/foriceberg"

cd "${GITHUB_ROOT}"

if [[ -n "${ICEBERG_PYTHON:-}" && -x "${ICEBERG_PYTHON}" ]]; then
  PYTHON_BIN="${ICEBERG_PYTHON}"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  PYTHON_BIN="python3"
fi

export ICEBERG_PYTHON="${PYTHON_BIN}"
echo "[train] python=${PYTHON_BIN}" >&2
exec "${PYTHON_BIN}" "${FORICEBERG_SCRIPT}/run_train_from_dataset.py" "$@"
