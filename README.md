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
| Structure annotation | `script/foriceberg/build_pubchem_top100_by_ecfp4_from_mgf.py` | Build PubChem top100 candidate CSV |
| Structure annotation | `script/foriceberg/run_iceberg_retrieval_eval_evaltest.py` | Candidate CSV + MGF retrieval eval |
| Structure annotation | `script/foriceberg/evaluate_test_set.py` | Evaluate existing `preds.hdf5` |
| Structure annotation | `script/foriceberg/process_new_dataset_from_mgf_with_parentmass.py` | Convert MGF to ICEBERG dataset |
| Structure annotation | `script/foriceberg/run_magma.py` | Run MAGMA fragmentation tree generation |
| Structure annotation | `script/foriceberg/run_retrain_ficeberg_split1_newloc.py` | Retrain F-ICEBERG split |
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

### 5.5 Retrieval Eval from Candidate CSV

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

### 5.6 Build PubChem Top100 Candidate CSV

Generate PubChem top100 candidates by ECFP4 similarity.

```bash
python script/foriceberg/build_pubchem_top100_by_ecfp4_from_mgf.py \
  --mgf "${EX}/F_ICEBERG_calibrated_test10.mgf" \
  --pubchem-hdf5 "${PUBCHEM}" \
  --out-csv "${EX}/pubchem_top100_test10.csv" \
  --max-topk 100
```

### 5.7 New Dataset and MAGMA

```bash
python script/foriceberg/process_new_dataset_from_mgf_with_parentmass.py --help
python script/foriceberg/run_magma.py --help
```

### 5.8 Retrain F-ICEBERG

Retraining requires the Zenodo retrain archive. Download it into `data/foriceberg/retrain/`, then extract it from the repository root.

```bash
cd /path/to/FMS2Lib
mkdir -p data/foriceberg/retrain

# Download these files from the Zenodo record:
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

## 6. Data Locations

| Path | Description |
|---|---|
| `data/testdataset/` | Formula prediction example data |
| `data/foriceberg/examples/` | ICEBERG example MGF, SusDat, predicted formulas, candidate CSV |
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
