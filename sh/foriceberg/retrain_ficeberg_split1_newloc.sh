#!/bin/bash
# Paths are resolved from the FMS2Lib repository root.
# Dataset: ``data/foriceberg/spec_datasets/F_ICEBERG_dataset/`` — see ../../README.md (Data Locations)
#
# Optional ICEBERG venv (full checkout): **does not** auto-source parent ``activate_env.sh``
# (that lives outside ``github/``). To load it from this wrapper:
#   export ICEBERG_SOURCE_ACTIVATE=1
# or run ``source …/activate_env.sh`` yourself before this script.
#
# Python: on clusters, ``python3`` may be a bare CVMFS/system build (no pandas/torch).
# Pick interpreter in this order:
#   1) ICEBERG_PYTHON (absolute path to your conda/venv ``python``)
#   2) CONDA_PREFIX/bin/python when conda is activated
#   3) VIRTUAL_ENV/bin/python after sourcing activate_env.sh / venv
#   4) PYTHON (if set)
#   5) ``python`` on PATH (often the venv after activation)
#   6) python3
set -euo pipefail
# FMS2Lib repo root (parent of sh/foriceberg/)
GITHUB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FORICEBERG_SCRIPT="${GITHUB_ROOT}/script/foriceberg"

ACTIVATE_SCRIPT="${ICEBERG_ACTIVATE_SCRIPT:-${GITHUB_ROOT}/../activate_env.sh}"
if [[ "${ICEBERG_SKIP_ACTIVATE:-0}" == "1" ]]; then
  :
elif [[ "${ICEBERG_SOURCE_ACTIVATE:-0}" == "1" && -f "${ACTIVATE_SCRIPT}" ]]; then
  echo "[retrain] source ${ACTIVATE_SCRIPT} (ICEBERG_SOURCE_ACTIVATE=1)" >&2
  # shellcheck disable=SC1090
  source "${ACTIVATE_SCRIPT}"
fi
cd "${GITHUB_ROOT}"

if [[ -n "${ICEBERG_PYTHON:-}" && -x "${ICEBERG_PYTHON}" ]]; then
  PYTHON_BIN="${ICEBERG_PYTHON}"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
elif [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="${PYTHON}"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  PYTHON_BIN="python3"
fi

export ICEBERG_PYTHON="${PYTHON_BIN}"
echo "[retrain] python=${PYTHON_BIN}" >&2
exec "${PYTHON_BIN}" "${FORICEBERG_SCRIPT}/run_retrain_ficeberg_split1_newloc.py" "$@"
