# FMS2Lib

FMS2Lib is an MS/MS spectral library of 35,000 low-fluorinated compounds for machine learning-assisted nontargeted analysis. This repository provides two workflow modules: formula prediction and structure annotation.

## 1. Installation

### Clone

```bash
git clone https://github.com/huiUofT/FMS2Lib.git
cd FMS2Lib
git lfs pull
```

### Environment

```bash
pip install -r requirements.txt
pip install -e .
```

For CUDA workflows, install `torch`, `dgl`, and `torch-geometric` wheels that match the local CUDA/Python environment.

### Environment Variables

```bash
export FMS2LIB=/path/to/FMS2Lib
cd "${FMS2LIB}"
export PYTHONPATH="${FMS2LIB}/src:${FMS2LIB}/src/foriceberg:${PYTHONPATH:-}"
```

### Optional: Download PubChem HDF5

```bash
python script/foriceberg/download_pubchem_hdf5.py
```

This downloads:

```bash
data/foriceberg/pubchem_formulae_inchikey.hdf5
```

Download it before running workflows that use `--pubchem-hdf5` or regenerate PubChem candidate CSVs.

## 2. Core Workflow: MS/MS to Structure Annotation

For nontargeted structure annotation from an experimental MS/MS spectrum, first predict the molecular formula with the formula ranker, then use the predicted formula to retrieve structures from PubChem/SusDat candidates.

```bash
EX=data/foriceberg/examples
PUBCHEM=data/foriceberg/pubchem_formulae_inchikey.hdf5
FORMULA_OUT=script/formula_prediction/results/formula_pred_test10
RANKER_MODEL=/path/to/model_ms1_ms2_ranker.joblib

# Step 1: MS/MS -> predicted molecular formula.
python script/formula_prediction/eval/run_eval_generate_and_predict.py \
  --mgf "${EX}/F_ICEBERG_calibrated_test10.mgf" \
  --mgf-only \
  --output-dir "${FORMULA_OUT}" \
  --model "${RANKER_MODEL}" \
  --no-mmap

# If RANKER_MODEL is model_ms1_ms2_ranker.joblib, this writes:
PRED_CSV="${FORMULA_OUT}/predictions_model_ms1_ms2_ranker.csv"

# Step 2: predicted formula + MS/MS -> structure annotation.
python script/foriceberg/blind_retrieval_from_pred_formula.py \
  --mgf "${EX}/F_ICEBERG_calibrated_test10.mgf" \
  --pred-csv "${PRED_CSV}" \
  --susdat "${EX}/susdat_F.csv" \
  --pubchem-hdf5 "${PUBCHEM}" \
  --out-dir script/foriceberg/results/blind_run1 \
  --gpu
```

The bundled example also includes a precomputed prediction CSV at `${EX}/predictions_model_ms1_ms2_ranker.csv`, which can be used directly for testing Step 2.

## 3. Script Overview

| Area | Script | Purpose |
|---|---|---|
| Formula prediction | `script/formula_prediction/calibration/clean.py` | Calibrate MGF and assign fragment formulas |
| Formula prediction | `script/formula_prediction/calibration/subformula.py` | Subformula assignment helper |
| Formula prediction | `script/formula_prediction/netural_loss/subformula.py` | Subformula assignment for neutral-loss workflow |
| Formula prediction | `script/formula_prediction/netural_loss/calculate_inter_fragment_nl.py` | Expand inter-fragment neutral losses |
| Formula prediction | `script/formula_prediction/netural_loss/nl_frequency_by_compound.py` | Count neutral losses per compound |
| Formula prediction | `script/formula_prediction/netural_loss/nl_analysis_pipeline.py` | End-to-end neutral-loss pipeline |
| Formula prediction | `script/formula_prediction/train/generate_training_data.py` | Generate MSBuddy feature joblibs |
| Formula prediction | `script/formula_prediction/train/train_ranker.py` | Train LightGBM formula ranker |
| Formula prediction | `script/formula_prediction/train/mgf_to_results.py` | MGF + formulas to features and ranker training |
| Formula prediction | `script/formula_prediction/train/pipeline_generate_and_train_ranker.py` | Training pipeline wrapper |
| Formula prediction | `script/formula_prediction/eval/generate_eval_joblibs.py` | Generate eval/inference feature joblibs |
| Formula prediction | `script/formula_prediction/eval/run_eval_generate_and_predict.py` | Generate eval features and run ranker prediction |
| Structure annotation | `script/foriceberg/download_pubchem_hdf5.py` | Download PubChem HDF5 |
| Structure annotation | `script/foriceberg/smiles_to_predicted_mgf.py` | SMILES to predicted MGF |
| Structure annotation | `script/foriceberg/run_predict_and_evaluate_testset.py` | Test labels to ICEBERG prediction and cosine eval |
| Structure annotation | `script/foriceberg/run_end2end_from_mgf_susdat_onepass.py` | MGF + SusDat candidates to Top-1 retrieval |
| Structure annotation | `script/foriceberg/blind_retrieval_from_pred_formula.py` | Predicted formula + MGF + PubChem/SusDat to blind retrieval |
| Structure annotation | `script/foriceberg/rank_pool_candidates_from_mgf.py` | Fixed candidate table + MGF to ranked SMILES CSV |
| Structure annotation | `script/foriceberg/build_pubchem_top100_by_ecfp4_from_mgf.py` | Build PubChem top100 candidate CSV |
| Structure annotation | `script/foriceberg/run_iceberg_retrieval_eval_evaltest.py` | Candidate CSV + MGF retrieval eval |
| Structure annotation | `script/foriceberg/evaluate_test_set.py` | Evaluate existing `preds.hdf5` |
| Structure annotation | `script/foriceberg/process_new_dataset_from_mgf_with_parentmass.py` | Convert MGF to ICEBERG dataset |
| Structure annotation | `script/foriceberg/sample_mgf_subset.py` | Deterministic MGF subset sampling |
| Structure annotation | `script/foriceberg/assign_subformulae.py` | Build `subformulae/no_subform.hdf5` |
| Structure annotation | `script/foriceberg/run_magma.py` | Run MAGMA fragmentation tree generation |
| Structure annotation | `script/foriceberg/run_train_from_dataset.py` | Train gen+inten on a prepared dataset |
| Structure annotation | `script/foriceberg/run_retrain_ficeberg_split1_newloc.py` | Retrain full F-ICEBERG split |
| Structure annotation | `script/foriceberg/finetune_gen_full.py` | Full fine-tune generation model from checkpoint |
| Structure annotation | `script/foriceberg/finetune_inten_partial_layers.py` | Partial-layer fine-tune intensity model |
| Structure annotation | `script/foriceberg/add_dag_intens.py` | Add/copy DAG intensities |
| Structure annotation | `script/foriceberg/foriceberg_paths.py` | Shared path helper |

## 4. Formula Prediction

Example inputs are under:

```bash
data/testdataset/
```

### 4.1 MGF Calibration

Calibrate an MGF using known precursor formulas and write a calibrated MGF.

```bash
python script/formula_prediction/calibration/clean.py \
  --mgf data/testdataset/F_ICEBERG_calibrated_test10.mgf \
  --formulas data/testdataset/test10_formulas.txt \
  --output scratch/calibrated_out.mgf \
  --ms1-tol 2 \
  --ms2-tol 5 \
  --msbuddy-dir src
```

### 4.2 Neutral-Loss Analysis

Run subformula assignment, expand inter-fragment neutral losses, and summarize neutral-loss frequency by compound.

```bash
python script/formula_prediction/netural_loss/nl_analysis_pipeline.py \
  --mgf data/testdataset/F_ICEBERG_calibrated_test10.mgf \
  --formulas data/testdataset/test10_formulas.txt \
  --meta-csv data/testdataset/test10_nl_meta.csv \
  --output-dir scratch/nl_out \
  --ms1-tol 2 \
  --ms2-tol 5 \
  --msbuddy-dir src
```

### 4.3 Train Formula Ranker

Generate MSBuddy features and train a LightGBM formula ranker.

```bash
python script/formula_prediction/train/mgf_to_results.py \
  --mgf data/testdataset/F_ICEBERG_calibrated_test10.mgf \
  --formulas data/testdataset/test10_formulas.txt \
  --output-dir scratch/train_out \
  --repo-root "${FMS2LIB}"
```

### 4.4 Evaluate Formula Ranker

Generate eval features and run a trained ranker.

```bash
python script/formula_prediction/eval/run_eval_generate_and_predict.py \
  --mgf data/testdataset/F_ICEBERG_calibrated_test10.mgf \
  --mgf-only \
  --output-dir scratch/eval_out \
  --model scratch/train_out/trained_models/model_ms1_ms2_ranker.joblib \
  --no-mmap
```

## 5. Structure Annotation

Example inputs are under:

```bash
EX=data/foriceberg/examples
PUBCHEM=data/foriceberg/pubchem_formulae_inchikey.hdf5
```

### 5.1 SMILES to Predicted MGF

Predict MS/MS spectra directly from SMILES.

```bash
python script/foriceberg/smiles_to_predicted_mgf.py \
  --smiles-file "${EX}/example_smiles.txt" \
  --out script/foriceberg/results/smiles_pred.mgf \
  --gpu
```

### 5.2 Predict and Evaluate Test Set

Run ICEBERG prediction on test labels and evaluate cosine similarity.

```bash
python script/foriceberg/run_predict_and_evaluate_testset.py \
  --dataset-labels data/foriceberg/spec_datasets/F_ICEBERG_dataset/testset_split_1_labels.tsv \
  --save-dir script/foriceberg/results/predictions_run1 \
  --output-dir script/foriceberg/results/evaluation_run1 \
  --dataset-name F_ICEBERG_dataset \
  --gpu
```

### 5.3 End-to-End Retrieval

Use MGF formulas and SusDat candidates, predict spectra for candidates, and evaluate Top-1 retrieval.

```bash
python script/foriceberg/run_end2end_from_mgf_susdat_onepass.py \
  --mgf "${EX}/mini_onepass_test.mgf" \
  --susdat "${EX}/susdat_F.csv" \
  --out-dir script/foriceberg/results/end2end_mini \
  --gpu
```

### 5.4 Blind Retrieval from Predicted Formula

Use predicted formulas to retrieve structures from PubChem/SusDat candidates.

```bash
python script/foriceberg/blind_retrieval_from_pred_formula.py \
  --mgf "${EX}/F_ICEBERG_calibrated_test10.mgf" \
  --pred-csv "${EX}/predictions_model_ms1_ms2_ranker.csv" \
  --susdat "${EX}/susdat_F.csv" \
  --pubchem-hdf5 "${PUBCHEM}" \
  --out-dir script/foriceberg/results/blind_run1 \
  --gpu
```

### 5.5 Fixed Candidate Pool Retrieval

Rank a fixed SMILES candidate table against an experimental MGF. This workflow does not insert a true SMILES from the MGF; it predicts spectra for candidates from the table and writes one ranked CSV.

Bundled mini example (1 spectrum with strong ICEBERG match, 3 candidates):

```bash
python script/foriceberg/rank_pool_candidates_from_mgf.py \
  --mgf "${EX}/pool_retrieval_mini.mgf" \
  --candidates "${EX}/pool_candidates_mini.csv" \
  --out-dir script/foriceberg/results/pool_retrieval_mini \
  --gen-checkpoint mode/foriceberg/generation/best.ckpt \
  --inten-checkpoint mode/foriceberg/intensity/best.ckpt \
  --gpu
```

The output is:

```text
script/foriceberg/results/pool_retrieval_mini/pool_candidate_ranking.csv
```

`--candidates` also accepts `.xlsx` / `.tsv` with a `SMILES` / `smile` / `candidate_smiles` column.

### 5.6 Retrieval Eval from Candidate CSV

Evaluate candidate ranking using a prebuilt candidate CSV and experimental MGF.

```bash
python script/foriceberg/run_iceberg_retrieval_eval_evaltest.py \
  --csv "${EX}/pubchem_top100_test10.csv" \
  --mgf "${EX}/F_ICEBERG_calibrated_test10.mgf" \
  --out-dir script/foriceberg/results/retrieval_mini \
  --gen-checkpoint mode/foriceberg/generation/best.ckpt \
  --inten-checkpoint mode/foriceberg/intensity/best.ckpt \
  --gpu
```

### 5.7 Build PubChem Top100 Candidate CSV

Generate PubChem top100 candidates by ECFP4 similarity.

```bash
python script/foriceberg/build_pubchem_top100_by_ecfp4_from_mgf.py \
  --mgf "${EX}/F_ICEBERG_calibrated_test10.mgf" \
  --pubchem-hdf5 "${PUBCHEM}" \
  --out-csv "${EX}/pubchem_top100_test10.csv" \
  --max-topk 100
```

### 5.8 Retrain F-ICEBERG

Full retrain follows the MGF → dataset → subformulae → MAGMA → generation → intensity workflow.

Use a GPU node with RDKit and CUDA-matched `torch` / `dgl` / `torch-geometric`.

```bash
cd /path/to/FMS2Lib
export FMS2LIB="$(pwd)"
export PYTHONPATH="${FMS2LIB}/src:${FMS2LIB}/src/foriceberg:${PYTHONPATH:-}"
export ICEBERG_PYTHON="$(which python)"

MGF=data/FMS2lib/F_ICEBERG.mgf
DATASET=F_ICEBERG_dataset
DATA_DIR=data/foriceberg/spec_datasets/${DATASET}
```

#### Path A — From MGF (complete pipeline)

##### Step 1 — Build dataset

```bash
python script/foriceberg/process_new_dataset_from_mgf_with_parentmass.py \
  --mgf-file "${MGF}" \
  --dataset-name "${DATASET}" \
  --seed 1
```

This writes `labels.tsv`, `spec_files.hdf5`, and `splits/split_1.tsv`. `HCD` is remapped to `Orbitrap` automatically.

##### Step 2 — Subformulae / intensity tables

```bash
python script/foriceberg/assign_subformulae.py \
  --data-dir "${DATA_DIR}" \
  --labels-file "${DATA_DIR}/labels.tsv" \
  --use-all \
  --output-dir-name no_subform.hdf5 \
  --num-workers 8
```

##### Step 3 — MAGMA trees

```bash
python script/foriceberg/run_magma.py \
  --spectra-dir "${DATA_DIR}/spec_files.hdf5" \
  --spec-labels "${DATA_DIR}/labels.tsv" \
  --output-dir "${DATA_DIR}/magma_outputs" \
  --max-peaks 50 \
  --ppm-diff 20 \
  --workers 32
```

`run_magma.py` also creates `magma_tree.hdf5` → `magma_tree_new.hdf5` for training.

##### Step 4 — Train generation + intensity models

One-shot (docx hyper-parameters):

```bash
bash sh/foriceberg/train_from_dataset.sh \
  --dataset-name "${DATASET}" \
  --gpu
```

Or the dedicated full-split driver (same stages, fixed `F_ICEBERG_dataset` paths):

```bash
bash sh/foriceberg/retrain_ficeberg_split1_newloc.sh
```

Manual stages (same hyper-parameters as the docx):

```bash
# 4a generation model
python -m ms_pred.dag_pred.train_gen \
  --dataset-name "${DATASET}" \
  --dataset-labels labels.tsv \
  --split-name split_1.tsv \
  --save-dir script/foriceberg/results/dag_${DATASET}/retrain \
  --magma-folder magma_outputs \
  --seed 1 --gpu --num-workers 8 --batch-size 32 --max-epochs 200 \
  --learning-rate 0.000996 --lr-decay-rate 0.7214 --dropout 0.2 \
  --mpnn-type GGNN --pe-embed-k 14 --pool-op avg --set-layers 2 \
  --hidden-size 512 --weight-decay 0.0 --layers 3 --root-encode gnn \
  --encode-forms --embed-adduct --embed-collision --embed-elem-group --add-hs

# 4b predict training DAGs
python -m ms_pred.dag_pred.predict_gen \
  --checkpoint-pth script/foriceberg/results/dag_${DATASET}/retrain/version_0/best.ckpt \
  --dataset-name "${DATASET}" \
  --dataset-labels labels.tsv \
  --split-name split_1.tsv \
  --save-dir script/foriceberg/results/dag_${DATASET}/retrain/preds_train_100 \
  --batch-size 8 --num-workers 8 --threshold 0.0 --max-nodes 100 --num-decoys 0 --gpu

# 4c merge intensities
python script/foriceberg/add_dag_intens.py \
  --pred-dag-folder script/foriceberg/results/dag_${DATASET}/retrain/preds_train_100/tree_preds.hdf5 \
  --true-dag-folder "${DATA_DIR}/subformulae/no_subform.hdf5" \
  --out-dag-folder script/foriceberg/results/dag_${DATASET}/retrain/preds_train_100_inten.hdf5 \
  --add-raw --num-workers 8

# 4d intensity model
python -m ms_pred.dag_pred.train_inten \
  --dataset-name "${DATASET}" \
  --dataset-labels labels.tsv \
  --split-name split_1.tsv \
  --magma-dag-folder script/foriceberg/results/dag_${DATASET}/retrain/preds_train_100_inten.hdf5 \
  --save-dir script/foriceberg/results/dag_inten_${DATASET}/retrain \
  --seed 1 --gpu --num-workers 8 --batch-size 32 --max-epochs 200 \
  --learning-rate 0.000736 --lr-decay-rate 0.825 --dropout 0.1 \
  --mpnn-type GGNN --pe-embed-k 0 --pool-op avg --hidden-size 256 \
  --weight-decay 1e-7 --set-layers 0 --frag-set-layers 3 --mlp-layers 1 \
  --gnn-layers 4 --loss-fn entropy --root-encode gnn \
  --include-unshifted-mz --binned-targs \
  --embed-adduct --embed-collision --embed-elem-group --encode-forms --add-hs \
  --sk-tau 0.05 --ppm-tol 20
```

Checkpoints:

```text
script/foriceberg/results/dag_F_ICEBERG_dataset/.../version_0/best.ckpt
script/foriceberg/results/dag_inten_F_ICEBERG_dataset/.../version_0/best.ckpt
```

#### Path B — Zenodo retrain archive (skip Steps 1–3)

If you already published prepared artifacts, download them and jump to training:

```bash
mkdir -p data/foriceberg/retrain

# Download from your Zenodo record:
# data/foriceberg/retrain/F_ICEBERG_retrain_artifacts.tar.gz
# data/foriceberg/retrain/F_ICEBERG_retrain_artifacts.tar.gz.sha256

echo "c1db0536f3b60bde49a99fadeec2f93bb1492b9c0acbcfc7c0a1d0bd2831de6b  data/foriceberg/retrain/F_ICEBERG_retrain_artifacts.tar.gz" | sha256sum -c -
tar -xzf data/foriceberg/retrain/F_ICEBERG_retrain_artifacts.tar.gz

export ICEBERG_PYTHON="$(which python)"
bash sh/foriceberg/retrain_ficeberg_split1_newloc.sh
```

The archive restores:

```text
data/foriceberg/spec_datasets/F_ICEBERG_dataset/labels.tsv
data/foriceberg/spec_datasets/F_ICEBERG_dataset/spec_files.hdf5
data/foriceberg/spec_datasets/F_ICEBERG_dataset/splits/split_1.tsv
data/foriceberg/spec_datasets/F_ICEBERG_dataset/magma_outputs/magma_tree.hdf5
data/foriceberg/spec_datasets/F_ICEBERG_dataset/subformulae/no_subform.hdf5
```

#### Optional — 100-spectrum smoke check

To verify the same Path A commands on a tiny subset, use `data/foriceberg/examples/F_ICEBERG_train100.mgf` and set `DATASET=F_ICEBERG_train100`. That run only validates the pipeline; it is not a production model.

```bash
python script/foriceberg/sample_mgf_subset.py \
  --mgf data/FMS2lib/F_ICEBERG.mgf \
  --out data/foriceberg/examples/F_ICEBERG_train100.mgf \
  --n 100 \
  --seed 1
```

## 6. Data Locations

| Path | Description |
|---|---|
| `data/testdataset/` | Formula prediction example data |
| `data/foriceberg/examples/` | ICEBERG example MGF, SusDat, predicted formulas, candidate CSV |
| `data/foriceberg/examples/F_ICEBERG_train100.mgf` | 100-spectrum training smoke MGF (seed=1) |
| `data/FMS2lib/F_ICEBERG.mgf` | Full F-ICEBERG MGF source for sampling / dataset builds |
| `data/foriceberg/spec_datasets/F_ICEBERG_dataset/` | ICEBERG dataset files |
| `data/foriceberg/retrain/` | Download location for the F-ICEBERG retrain archive |
| `mode/foriceberg/generation/best.ckpt` | ICEBERG generation checkpoint |
| `mode/foriceberg/intensity/best.ckpt` | ICEBERG intensity checkpoint |
| `data/foriceberg/pubchem_formulae_inchikey.hdf5` | Downloaded PubChem HDF5 |

## 7. Notes

- `torch`, `dgl`, and `torch-geometric` must match the local CUDA/Python environment.
- PubChem HDF5 must be downloaded before workflows using `--pubchem-hdf5`.
- MsBuddy database files download automatically on first use.
- Use `python <script.py> --help` for full options.

## 8. Fine-tune (100-spectrum example)

Smoke fine-tune on `F_ICEBERG_train100`. First prepare the dataset with the optional 100-spectrum path in §5.8 (MGF → dataset → subformulae → MAGMA → `train_from_dataset.sh`), or reuse an existing `data/foriceberg/spec_datasets/F_ICEBERG_train100/`.

```bash
cd /path/to/FMS2Lib
export FMS2LIB="$(pwd)"
export PYTHONPATH="${FMS2LIB}/src:${FMS2LIB}/src/foriceberg:${PYTHONPATH:-}"

DATASET=F_ICEBERG_train100
DATA_DIR=data/foriceberg/spec_datasets/${DATASET}
# Intensity DAG from §5.8 train_from_dataset (adjust if your run name differs):
INTEN_DAG=script/foriceberg/results/dag_${DATASET}/${DATASET}_rnd1_docx/preds_train_100_inten.hdf5
```

### 8.1 Generation model (full fine-tune)

```bash
python script/foriceberg/finetune_gen_full.py \
  --pretrained-checkpoint mode/foriceberg/generation/best.ckpt \
  --dataset-name "${DATASET}" \
  --split-name split_1.tsv \
  --magma-folder magma_outputs \
  --save-dir script/foriceberg/results/gen_finetune_${DATASET} \
  --learning-rate 1e-4 \
  --max-epochs 50 \
  --batch-size 32 \
  --num-workers 8 \
  --gpu
```

Output checkpoint:

```text
script/foriceberg/results/gen_finetune_F_ICEBERG_train100/version_0/best.ckpt
```

### 8.2 Intensity model (partial-layer fine-tune)

Requires `--magma-dag-folder` (predicted DAGs with intensities), e.g. `preds_train_100_inten.hdf5` from §5.8.

```bash
# List IntenGNN modules / presets
python script/foriceberg/finetune_inten_partial_layers.py \
  --pretrained-checkpoint mode/foriceberg/intensity/best.ckpt \
  --save-dir /tmp/inten_list \
  --list-modules

# Example: fine-tune MLP + output head
python script/foriceberg/finetune_inten_partial_layers.py \
  --pretrained-checkpoint mode/foriceberg/intensity/best.ckpt \
  --dataset-name "${DATASET}" \
  --split-name split_1.tsv \
  --magma-dag-folder "${INTEN_DAG}" \
  --preset mlp_head \
  --save-dir script/foriceberg/results/inten_finetune_${DATASET}_mlp_head \
  --learning-rate 1e-4 \
  --max-epochs 50 \
  --batch-size 32 \
  --num-workers 8 \
  --gpu
```

Other presets: `before_mlp` (Transformer only), `head`, `gnn`, `all`. See `script/foriceberg/README_finetune_inten_partial_layers.md`.

Output checkpoint:

```text
script/foriceberg/results/inten_finetune_F_ICEBERG_train100_mlp_head/version_0/best.ckpt
```

This 100-spectrum fine-tune is for pipeline checks only, not a production model.
