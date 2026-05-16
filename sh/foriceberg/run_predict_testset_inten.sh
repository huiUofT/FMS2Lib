#!/usr/bin/env bash
# Joint prediction (generation + intensity) on rows from a labels TSV.
# Run from anywhere; cwd is set to the FMS2Lib repo root.
set -euo pipefail

FMS2LIB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FORICEBERG_SRC="${FMS2LIB_ROOT}/src/foriceberg"
cd "${FMS2LIB_ROOT}"

: "${GEN_CHECKPOINT:?set GEN_CHECKPOINT (gen .ckpt path)}"
: "${INTEN_CHECKPOINT:?set INTEN_CHECKPOINT (inten .ckpt path)}"
: "${TEST_LABELS:?set TEST_LABELS (labels.tsv path, e.g. data/foriceberg/spec_datasets/.../testset_split_1_labels.tsv)}"
: "${SAVE_DIR:?set SAVE_DIR (output dir; will contain preds.hdf5)}"

export PYTHONPATH="${FORICEBERG_SRC}${PYTHONPATH:+:${PYTHONPATH}}"

exec python "${FORICEBERG_SRC}/ms_pred/dag_pred/predict_smis_trained_inten.py" \
  --gen-checkpoint "${GEN_CHECKPOINT}" \
  --inten-checkpoint "${INTEN_CHECKPOINT}" \
  --dataset-labels "${TEST_LABELS}" \
  --save-dir "${SAVE_DIR}" \
  --sparse-out --sparse-k 100 \
  --num-workers "${NUM_WORKERS:-4}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --max-nodes "${MAX_NODES:-5000}" \
  --threshold "${THRESHOLD:-0.05}" \
  --adduct-shift \
  --gpu
