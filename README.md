# FMS2Lib

MS/MS workflows for fluorinated compounds: **formula ranker** (`script/formula_prediction/`, MSBuddy) and **ICEBERG DAG prediction** (`script/foriceberg/`, `src/foriceberg/ms_pred`).

## Setup

```bash
git clone https://github.com/huiUofT/FMS2Lib.git
cd FMS2Lib
git lfs pull    # spec_files.hdf5

pip install -r requirements.txt
pip install -e .   # msbuddy from src/
```

**ICEBERG** (each new shell):

```bash
export FMS2LIB=/path/to/FMS2Lib
cd "${FMS2LIB}"
export PYTHONPATH="${FMS2LIB}/src/foriceberg:${PYTHONPATH:-}"
```

**Formula ranker** (add MSBuddy to path):

```bash
export PYTHONPATH="${FMS2LIB}/src:${PYTHONPATH:-}"
```

GPU training needs CUDA-matched `torch` / `dgl` / PyG (see `requirements.txt`). MsBuddy `.joblib` DBs download on first use (~416 MB).

**Bundled data** (`data/foriceberg/spec_datasets/F_ICEBERG_dataset/`): `labels.tsv`, `spec_files.hdf5`, `testset_split_1_labels.tsv`. For retrain, add locally: `splits/split_1.tsv`, `magma_outputs/magma_tree.hdf5`, `subformulae/no_subform.hdf5` (generate with scripts below or symlink a full dataset tree).

Default checkpoints: `mode/foriceberg/generation/best.ckpt`, `mode/foriceberg/intensity/best.ckpt`.

---

## ICEBERG (`script/foriceberg/`)

Run from **`${FMS2LIB}`**. Flags: `python script/foriceberg/<script>.py --help`.

| Script | Purpose |
|--------|---------|
| `process_new_dataset_from_mgf_with_parentmass.py` | MGF → `labels.tsv`, `spec_files.hdf5`, `splits/split_<seed>.tsv` |
| `run_magma.py` | `spec_files.hdf5` + `labels.tsv` → `magma_outputs/magma_tree.hdf5` |
| `run_retrain_ficeberg_split1_newloc.py` | F_ICEBERG split_1 retrain (gen + inten) |
| `run_predict_and_evaluate_testset.py` | Test labels → `preds.hdf5` → cosine evaluation |
| `evaluate_test_set.py` | Cosine eval only (given `preds.hdf5`) |
| `smiles_to_predicted_mgf.py` | SMILES → predicted MS/MS MGF |
| `run_end2end_from_mgf_susdat_onepass.py` | MGF + SusDat CSV → predict → Top-1 eval |
| `run_iceberg_retrieval_eval_evaltest.py` | Candidate CSV + MGF → retrieval eval |
| `blind_retrieval_from_pred_formula.py` | Predicted formula + MGF → blind retrieval |
| `build_pubchem_top100_by_ecfp4_from_mgf.py` | MGF + PubChem HDF5 → top-K CSV |
| `add_dag_intens.py` | Copy intensities from gold DAGs into predicted DAG HDF5 |

**Shell** (`sh/foriceberg/`): `retrain_ficeberg_split1_newloc.sh`, `run_predict_testset_inten.sh`, `run_evaluate_testset_cosine.sh`.

### Examples

**Predict + evaluate** (split 1 test set):

```bash
python script/foriceberg/run_predict_and_evaluate_testset.py \
  --dataset-labels data/foriceberg/spec_datasets/F_ICEBERG_dataset/testset_split_1_labels.tsv \
  --save-dir script/foriceberg/results/predictions_run1 \
  --output-dir script/foriceberg/results/evaluation_run1 \
  --dataset-name F_ICEBERG_dataset \
  --gpu
```

**Retrain** (needs full dataset artifacts):

```bash
bash sh/foriceberg/retrain_ficeberg_split1_newloc.sh
# or: python script/foriceberg/run_retrain_ficeberg_split1_newloc.py
```

**New dataset from MGF**:

```bash
python script/foriceberg/process_new_dataset_from_mgf_with_parentmass.py \
  --mgf-file /path/to/input.mgf \
  --dataset-name my_dataset
```

**MAGMA trees** (after `spec_files.hdf5` exists):

```bash
python script/foriceberg/run_magma.py --help
```

**Eval only** (if `preds.hdf5` already exists):

```bash
python script/foriceberg/evaluate_test_set.py \
  --pred-file script/foriceberg/results/predictions_run1/preds.hdf5 \
  --test-labels data/foriceberg/spec_datasets/F_ICEBERG_dataset/testset_split_1_labels.tsv \
  --dataset-name F_ICEBERG_dataset \
  --output-dir script/foriceberg/results/evaluation_run1
```

---

## Formula ranker (`script/formula_prediction/`)

```bash
export PYTHONPATH="${FMS2LIB}/src:${PYTHONPATH:-}"
python script/formula_prediction/calibration/clean.py --help
python script/formula_prediction/train/mgf_to_results.py --help
python script/formula_prediction/eval/run_eval_generate_and_predict.py --help
```

Test MGFs: `data/testdataset/`. Use `--repo-root "${FMS2LIB}"` and paths under `${FMS2LIB}/...` for self-contained runs.
