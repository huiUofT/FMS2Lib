# FMS2Lib

FMS2Lib is an MS/MS spectral library of approximately 35,000 low-fluorinated compounds for machine learning-assisted nontargeted analysis. This repository provides two workflow modules: formula prediction and structure annotation.

**Our article:** [FMS2Lib preprint on ChemRxiv](https://doi.org/10.26434/chemrxiv.15010212/v1).

**License:** original FMS2Lib software and documentation are available under the [MIT License](LICENSE). Third-party code retains its original licenses; see [third-party code and licenses](#third-party-code-and-licenses) for license scope.

### Code sources and attribution

We directly use code from the following projects in these workflows:

- **ICEBERG**, developed by Goldman et al., from [samgoldman97/ms-pred](https://github.com/samgoldman97/ms-pred). The code is included under `src/foriceberg/ms_pred/` and is used for spectrum prediction and structure annotation. FMS2Lib provides workflow scripts and retrained models using the project dataset.
- **MSBuddy**, developed by Xing et al., from [Philipbear/msbuddy](https://github.com/Philipbear/msbuddy), with the related [HuanLab/BUDDY](https://github.com/HuanLab/BUDDY) project referenced in the source headers. The Python code is included under `src/msbuddy/` and is used for candidate molecular formula generation, subformula assignment, and feature generation for formula ranking.

ICEBERG and MSBuddy are third-party implementations; their original methods and code are credited to their respective authors. The bundled files include upstream code and local adaptations, so they should not be assumed to be an exact copy of the latest upstream release. See [third-party code and licenses](#third-party-code-and-licenses) for license texts and local modification notices.

The full MGF contains **63,046 spectra**: 34,686 entries with fluorine in their SMILES and 28,360 without fluorine. The approximately 35,000 figure refers to the fluorinated subset, not the entire training input. See [data and model provenance](docs/DATA_AND_MODELS.md) for file availability and validation limits.

### Third-party code and licenses

| Component | Bundled location | Upstream source | License copy |
|---|---|---|---|
| ICEBERG / ms-pred, Goldman et al. | `src/foriceberg/ms_pred/` and adapted workflow scripts | [samgoldman97/ms-pred](https://github.com/samgoldman97/ms-pred) | [MIT, copyright 2023 Samuel Goldman](LICENSES/ICEBERG-MIT.txt) |
| MSBuddy, Xing et al. | `src/msbuddy/` | [Philipbear/msbuddy](https://github.com/Philipbear/msbuddy) | [Apache License 2.0](LICENSES/MSBuddy-Apache-2.0.txt); retain the existing Shipei Xing copyright notices |
| Microsoft Graphormer-derived algorithm | `src/foriceberg/ms_pred/massformer_pred/massformer_code/algos2.pyx` and its generated artifacts | [microsoft/Graphormer](https://github.com/microsoft/Graphormer) | [MIT, Microsoft Corporation](LICENSES/Graphormer-MIT.txt), consistent with the bundled source header |
| OGB preprocessing excerpt | `gf_preprocess` in `src/foriceberg/ms_pred/massformer_pred/massformer_code/gf_data_utils.py` | [snap-stanford/ogb](https://github.com/snap-stanford/ogb), as linked in the function docstring | [MIT, copyright 2019 OGB Team](LICENSES/OGB-MIT.txt) |

The MSBuddy source headers also reference the related [HuanLab/BUDDY](https://github.com/HuanLab/BUDDY) project. The Python files in `src/msbuddy/` identify MSBuddy and retain Apache-2.0 notices; the related project's license is not substituted for those notices.

The accompanying license copies supplement the existing source notices. The exact upstream commits used for the original code imports were not recorded.

Source headers identify adapted workflows and local model retraining. Upstream author information and copyright notices are retained. A syntax correction to the `selected_evs` dictionary comprehension in `retrieval_benchmark_torchmetrics.py` is noted in its source header.

Original FMS2Lib software and documentation contributions are licensed under the [MIT License](LICENSE). Third-party code retains its original copyright notices and licenses listed above. The MIT license for original FMS2Lib software does not replace those licenses or assign new reuse terms to third-party data, externally downloaded assets, or model weights. See [data and model provenance](docs/DATA_AND_MODELS.md) for source information and asset scope.

When citing the workflows, credit the relevant upstream software as well as the [FMS2Lib article](https://doi.org/10.26434/chemrxiv.15010212/v1).

## 1. Installation

### Clone

```bash
git clone https://github.com/huiUofT/FMS2Lib.git
cd FMS2Lib
git lfs install
git lfs pull
```

### Environment

Install Git LFS before cloning or pulling the HDF5 asset. See the [validated CPU environments and results](docs/VALIDATION.md) before installing dependencies. The following is the general dependency specification; its open-ended version ranges do not select the exact environments used in validation. Run these commands from the repository root, in a separate Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

This repository uses a source-directory layout without Python packaging metadata. Set `PYTHONPATH` as shown below before running the workflows.

Small-scale CPU inference was validated with Python 3.11. The formula environment used **LightGBM 4.5.0 and scikit-learn 1.5.2**; LightGBM 4.5.0 with scikit-learn 1.8.0 failed during prediction. The ICEBERG environment used **PyTorch 2.5.1+cpu, DGL 1.1.3**, and PyG extensions built for PyTorch 2.5 CPU. Both bundled checkpoints record **Lightning 2.6.1**; the tested **Lightning 1.9.5** loaded them with a version warning and completed the CPU examples. Full training and GPU compatibility have not been validated. `requirements.txt` is a general specification rather than a tested lock file. See the [separate CPU profiles](docs/VALIDATION.md#environment-records); a combined environment has not been verified.

For CUDA workflows, install `torch`, `dgl`, and `torch-geometric` wheels that match the local CUDA/Python environment. Most structure annotation examples below request a GPU explicitly; remove `--gpu` for CPU execution. The validation commands in the linked report use CPU execution and explicitly limit the candidate count for the end-to-end example.

Some auxiliary modules use packages not explicitly listed in the requirements, including `pathos` (parallel helpers), `cairosvg` (PDF rendering), `ipdb` (legacy modules), and `platformdirs` / `pubchempy` (elucidation utilities). Alternate XGBoost model loading requires `xgboost`. The fixed candidate-pool script parses `.xlsx` inputs using the Python standard library and does not require `openpyxl`. These optional or workflow-specific dependencies have not been added to the existing environment specification.

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

Download it before running workflows that use `--pubchem-hdf5` or regenerate PubChem candidate CSVs. This external PubChem database is separate from the bundled training spectra; there is no FMS2Lib retraining archive to download. See [data and model provenance](docs/DATA_AND_MODELS.md) for the source URL and download limitations.

## 2. Core Workflow: MS/MS to Structure Annotation

For nontargeted structure annotation from an experimental MS/MS spectrum, first predict the molecular formula with the formula ranker, then use the predicted formula to retrieve structures from PubChem/SusDat candidates.

```bash
EX=data/foriceberg/examples
PUBCHEM=data/foriceberg/pubchem_formulae_inchikey.hdf5
FORMULA_OUT=script/formula_prediction/results/formula_pred_test10
RANKER_MODEL=mode/model_ms1_ms2_ranker.joblib

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

The bundled example also includes a precomputed prediction CSV at `${EX}/predictions_model_ms1_ms2_ranker.csv`, which can be used directly for testing Step 2. For this ten-spectrum example, the bundled SusDat table has no candidates matching the predicted formulas, so the PubChem input is needed to obtain structure annotations. If all spectra lack candidates, the current script writes `skipped_spectra.csv` and exits with status 0 without producing the annotation CSV; inspect the outputs as well as the exit status.

Run commands from the repository root after setting the environment variables in §1. The formula step writes the prediction CSV above; the structure step writes candidate and annotation outputs under the requested `--out-dir`. The examples describe expected file locations, not independently reproduced article metrics.

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

The ten-spectrum input below illustrates the interface. Formula-stratified splitting can leave small strata without training examples, so this is not a validated training benchmark or a replacement for a full training dataset.

```bash
python script/formula_prediction/train/mgf_to_results.py \
  --mgf data/testdataset/F_ICEBERG_calibrated_test10.mgf \
  --formulas data/testdataset/test10_formulas.txt \
  --output-dir scratch/train_out \
  --repo-root "${FMS2LIB}"
```

### 4.4 Evaluate Formula Ranker

Generate eval features and run a trained ranker.

The command below expects a model produced by §4.3. To use the bundled model instead, set `--model mode/model_ms1_ms2_ranker.joblib`.

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

This benchmark uses the known true SMILES to select similar candidates and explicitly includes the true structure. It can produce up to 101 rows per spectrum when the true structure is not already among the selected 100 candidates. Its results describe ranking in this constructed candidate set. For retrieval using predicted formulas without using the true SMILES to build candidates, see §5.4.

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

MGF=data/FMS2Lib/FMS2Lib.mgf
DATASET=F_ICEBERG_dataset
DATA_DIR=data/foriceberg/spec_datasets/${DATASET}
```

#### Prepare training inputs from the bundled MGF

The repository provides `labels.tsv`, `spec_files.hdf5`, and `testset_split_1_labels.tsv`. The split TSV, subformula tables, and MAGMA trees are not bundled; generate them with Steps 1–3 before training. No prepared retraining archive is available. Step 1 writes the dataset files into `DATA_DIR`, including the already bundled labels and spectra; use a separate dataset name if you need to keep those files untouched during a new run.

##### Step 1 — Build dataset

```bash
python script/foriceberg/process_new_dataset_from_mgf_with_parentmass.py \
  --mgf-file "${MGF}" \
  --dataset-name "${DATASET}" \
  --seed 1
```

This writes `labels.tsv`, `spec_files.hdf5`, and `splits/split_1.tsv`. `HCD` is remapped to `Orbitrap` automatically.

The splitter stratifies spectrum rows by fluorine count with seed 1; it does not group by molecular connectivity. The bundled test labels share no complete InChIKeys with the remaining labels, but 14 InChIKey connectivity blocks occur in both sets. The source MGF has no collision-energy metadata, so preprocessing fills the value with `30.0`; the bundled labels use that value throughout. This is a preprocessing default, not independently verified acquisition metadata. Rebuilding a split with the supplied code has not been checked against the split used to train the bundled checkpoints.

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

One-shot using the driver's configured hyperparameters:

```bash
bash sh/foriceberg/train_from_dataset.sh \
  --dataset-name "${DATASET}" \
  --gpu
```

Or the dedicated full-split driver (same stages, fixed `F_ICEBERG_dataset` paths):

```bash
bash sh/foriceberg/retrain_ficeberg_split1_newloc.sh
```

Manual stages with explicit hyperparameters:

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

### 5.9 HF Tag Classifier (ported from `hf-predict`)

The HF binary tag training scripts are available at:

```text
script/foriceberg/hf_predict/train_tag_graph_transformer_hf.py
script/foriceberg/hf_predict/smiles_graph_transformer_hf.py
```

Training CSV is stored at:

```text
data/foriceberg/hf_predict/tag.csv
```

Run retraining with:

```bash
cd "${FMS2LIB}/script/foriceberg/hf_predict"
python train_tag_graph_transformer_hf.py \
  --train_csv "${FMS2LIB}/data/foriceberg/hf_predict/tag.csv" \
  --out_dir "${FMS2LIB}/script/foriceberg/results/hf_predict/output_g3_t2_bs200" \
  --epochs 30 \
  --batch_size 200 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --dropout 0.1 \
  --hidden_dim 192 \
  --fusion_hidden 256 \
  --num_layers 5 \
  --ggnn_layers 3 \
  --transformer_layers 2 \
  --num_heads 4 \
  --col_emb_dim 24 \
  --val_frac 0.1 \
  --test_frac 0.1 \
  --num_workers 4 \
  --seed 2026 \
  --split_mode scaffold \
  --use_class_weights
```

#### Optional — 100-spectrum smoke check

To try the same preprocessing and training commands on a small subset, use `data/foriceberg/examples/F_ICEBERG_train100.mgf` and set `DATASET=F_ICEBERG_train100`. This input is intended for pipeline checks, not production model training; full training on this input has not been validated.

```bash
python script/foriceberg/sample_mgf_subset.py \
  --mgf data/FMS2Lib/FMS2Lib.mgf \
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
| `data/FMS2Lib/FMS2Lib.mgf` | Bundled full MGF source: 63,046 spectra, including 34,686 fluorinated entries |
| `data/foriceberg/spec_datasets/F_ICEBERG_dataset/` | Bundled labels, spectra (Git LFS), and test labels; other training inputs are generated in §5.8 |
| `mode/model_ms1_ms2_ranker.joblib` | Bundled molecular formula ranker |
| `mode/foriceberg/generation/best.ckpt` | ICEBERG generation checkpoint |
| `mode/foriceberg/intensity/best.ckpt` | ICEBERG intensity checkpoint |
| `data/foriceberg/pubchem_formulae_inchikey.hdf5` | External PubChem HDF5; download separately when needed |

For per-file sources, generated inputs, and provenance still to be confirmed, see [data and model provenance](docs/DATA_AND_MODELS.md).

## 7. Notes

- `torch`, `dgl`, and `torch-geometric` must match the local CUDA/Python environment.
- PubChem HDF5 must be downloaded before workflows using `--pubchem-hdf5`.
- MSBuddy database files download automatically when initialized; network access is required if they are not cached.
- Use `python <script.py> --help` for full options.
- The historical directory names `mode/` and `netural_loss/` are retained because existing scripts and commands use them.
- Default data and output paths are resolved relative to the repository. Command-line options override those defaults.
- CPU validation covered the ten-spectrum formula example, a repeated three-candidate ICEBERG example, and a ten-spectrum end-to-end run capped at three candidates per query. GPU execution, default 400-candidate retrieval, full retraining, and article metrics were not validated; see [validation details](docs/VALIDATION.md).
- Original FMS2Lib software and documentation use the [MIT License](LICENSE); see [third-party code and licenses](#third-party-code-and-licenses) for retained third-party terms and asset scope.
- The optional MassFormer extension includes a platform-specific binary; see [build instructions](docs/MASSFORMER_EXTENSION.md) before using it on a different platform or Python version.

## 8. Fine-tune (100-spectrum example)

Smoke fine-tune on `F_ICEBERG_train100`. First prepare the dataset with the optional 100-spectrum example in §5.8 (MGF → dataset → subformulae → MAGMA → `train_from_dataset.sh`), or reuse an existing `data/foriceberg/spec_datasets/F_ICEBERG_train100/`.

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

Other presets: `before_mlp` (Transformer only; alias `transformer`), `head`, `gnn`, `all`. The module lists are documented in the source header and `--help` of [`finetune_inten_partial_layers.py`](script/foriceberg/finetune_inten_partial_layers.py).

Output checkpoint:

```text
script/foriceberg/results/inten_finetune_F_ICEBERG_train100_mlp_head/version_0/best.ckpt
```

This 100-spectrum fine-tune is for pipeline checks only, not a production model.
