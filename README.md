# FMS2Lib

MS/MS workflows for fluorinated compounds: **formula ranker** (`script/formula_prediction/`, MSBuddy) and **ICEBERG DAG prediction** (`script/foriceberg/`, `src/foriceberg/ms_pred`).

## Setup

```bash
git clone https://github.com/huiUofT/FMS2Lib.git
cd FMS2Lib
git lfs pull    # spec_files.hdf5

pip install -r requirements.txt
pip install -e .
```

```bash
export FMS2LIB=/path/to/FMS2Lib
cd "${FMS2LIB}"
export PYTHONPATH="${FMS2LIB}/src/foriceberg:${PYTHONPATH:-}"
```

GPU training needs CUDA-matched `torch` / `dgl` / PyG. MsBuddy `.joblib` DBs download on first use (~416 MB).

**Dataset in git** (`data/foriceberg/spec_datasets/F_ICEBERG_dataset/`): `labels.tsv`, `spec_files.hdf5` (LFS), `testset_split_1_labels.tsv`. Retrain also needs local `splits/split_1.tsv`, `magma_outputs/magma_tree.hdf5`, `subformulae/no_subform.hdf5`.

**Small examples** (`data/foriceberg/examples/`): MGF, SusDat, pred-formula CSV, mini retrieval CSV — see `data/foriceberg/examples/README.md`.

Checkpoints: `mode/foriceberg/generation/best.ckpt`, `mode/foriceberg/intensity/best.ckpt`.

---

## ICEBERG scripts

Run from `${FMS2LIB}`. Help: `python script/foriceberg/<script>.py --help`.

| Script | Purpose |
|--------|---------|
| `run_predict_and_evaluate_testset.py` | Test labels → predict → cosine eval |
| `run_retrain_ficeberg_split1_newloc.py` | Split_1 retrain |
| `process_new_dataset_from_mgf_with_parentmass.py` | MGF → dataset folder |
| `run_magma.py` | MAGMA fragmentation trees |
| `smiles_to_predicted_mgf.py` | SMILES → predicted MGF |
| `run_end2end_from_mgf_susdat_onepass.py` | MGF + SusDat → Top-1 eval |
| `run_iceberg_retrieval_eval_evaltest.py` | Candidate CSV + MGF → retrieval eval |
| `blind_retrieval_from_pred_formula.py` | Pred formula CSV + MGF → blind retrieval |
| `build_pubchem_top100_by_ecfp4_from_mgf.py` | MGF + PubChem HDF5 → candidate CSV |
| `evaluate_test_set.py` | Eval only (`preds.hdf5`) |

### Examples (bundled paths)

```bash
EX=data/foriceberg/examples
```

**Predict + evaluate** (official test split):

```bash
python script/foriceberg/run_predict_and_evaluate_testset.py \
  --dataset-labels data/foriceberg/spec_datasets/F_ICEBERG_dataset/testset_split_1_labels.tsv \
  --save-dir script/foriceberg/results/predictions_run1 \
  --output-dir script/foriceberg/results/evaluation_run1 \
  --dataset-name F_ICEBERG_dataset \
  --gpu
```

**SMILES → MGF**:

```bash
python script/foriceberg/smiles_to_predicted_mgf.py \
  --smiles-file "${EX}/example_smiles.txt" \
  --out script/foriceberg/results/smiles_pred.mgf \
  --gpu
```

**End2end** (mini MGF + SusDat):

```bash
python script/foriceberg/run_end2end_from_mgf_susdat_onepass.py \
  --mgf "${EX}/mini_onepass_test.mgf" \
  --susdat "${EX}/susdat_F.csv" \
  --out-dir script/foriceberg/results/end2end_mini \
  --gpu
```

**Blind retrieval** (test10 MGF + ranker pred formulas + SusDat):

```bash
python script/foriceberg/blind_retrieval_from_pred_formula.py \
  --mgf "${EX}/F_ICEBERG_calibrated_test10.mgf" \
  --pred-csv "${EX}/predictions_model_ms1_ms2_ranker.csv" \
  --susdat "${EX}/susdat_F.csv" \
  --out-dir script/foriceberg/results/blind_run1 \
  --gpu
```

> **Note:** Optional `--pubchem-hdf5` adds PubChem candidates; the HDF5 file is **not** in this repo (large). How to obtain or ship it — **TBD**.

**Retrieval eval** (mini candidate CSV + test10 MGF):

```bash
python script/foriceberg/run_iceberg_retrieval_eval_evaltest.py \
  --csv "${EX}/pubchem_top100_test10_mini.csv" \
  --mgf "${EX}/F_ICEBERG_calibrated_test10.mgf" \
  --out-dir script/foriceberg/results/retrieval_mini \
  --gen-checkpoint mode/foriceberg/generation/best.ckpt \
  --inten-checkpoint mode/foriceberg/intensity/best.ckpt \
  --gpu
```

> **Note:** `build_pubchem_top100_by_ecfp4_from_mgf.py` (MGF + PubChem-by-formula HDF5 → candidate CSV) requires a large PubChem HDF5 that is **not** in this repo. Example command and data source — **TBD**.

**Retrain** / **MAGMA** / **new dataset**: `bash sh/foriceberg/retrain_ficeberg_split1_newloc.sh` · `python script/foriceberg/run_magma.py --help` · `process_new_dataset_from_mgf_with_parentmass.py --mgf-file …`

---

## Formula ranker

```bash
export PYTHONPATH="${FMS2LIB}/src:${PYTHONPATH:-}"
python script/formula_prediction/calibration/clean.py --help
```

Test data: `data/testdataset/`.
