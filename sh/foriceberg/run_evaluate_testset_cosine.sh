#!/usr/bin/env bash
# Cosine evaluation (dist-bin style) on preds.hdf5 vs experimental spectra from labels.
# Expects predict step to have written "${SAVE_DIR}/preds.hdf5".
set -euo pipefail

FMS2LIB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FORICEBERG_SRC="${FMS2LIB_ROOT}/src/foriceberg"
FORICEBERG_SCRIPT="${FMS2LIB_ROOT}/script/foriceberg"
cd "${FMS2LIB_ROOT}"

: "${SAVE_DIR:?set SAVE_DIR (same as predict step; reads preds.hdf5 here)}"
: "${TEST_LABELS:?set TEST_LABELS (same labels TSV as predict step)}"
: "${OUTPUT_DIR:?set OUTPUT_DIR (writes evaluation_results.tsv and evaluation_stats.json)}"
: "${DATASET_NAME:?set DATASET_NAME (for data/foriceberg/spec_datasets/<name>/spec_files.hdf5 etc.)}"

export PYTHONPATH="${FORICEBERG_SRC}${PYTHONPATH:+:${PYTHONPATH}}"

exec python "${FORICEBERG_SCRIPT}/evaluate_test_set.py" \
  --pred-file "${SAVE_DIR}/preds.hdf5" \
  --test-labels "${TEST_LABELS}" \
  --output-dir "${OUTPUT_DIR}" \
  --dataset-name "${DATASET_NAME}" \
  --num-bins "${NUM_BINS:-15000}" \
  --upper-limit "${UPPER_LIMIT:-1500}"
