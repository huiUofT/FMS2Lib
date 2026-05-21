# `sh/foriceberg/` — shell entrypoints (FMS2Lib layout)

Wrappers set **`cwd`** to the **FMS2Lib repo root** and **`PYTHONPATH`** to **`src/foriceberg`**. See **`../../README.md`** (ICEBERG section).

## `retrain_ficeberg_split1_newloc.sh`

Runs **`script/foriceberg/run_retrain_ficeberg_split1_newloc.py`** with:

- `cwd` = **`github/`**
- `PYTHONPATH` = **`github/src`** (vendored `ms_pred`)
- Data = **`data/foriceberg/spec_datasets/F_ICEBERG_dataset/`** (see **`../../README.md`** → ICEBERG section)
- Checkpoints / outputs = **`script/foriceberg/results/dag_*_retrain/...`**

**ICEBERG checkout:** activate the venv yourself (nothing outside ``github/`` is sourced by default):

```bash
cd /path/to/FMS2Lib
source activate_env.sh
cd github
bash sh/retrain_ficeberg_split1_newloc.sh
```

To **auto-source** a parent ``activate_env.sh`` from the wrapper (optional convenience when nested in the full repo):

```bash
export ICEBERG_SOURCE_ACTIVATE=1
bash sh/retrain_ficeberg_split1_newloc.sh
```

Paths: ``ICEBERG_ACTIVATE_SCRIPT`` overrides which script to source. ``ICEBERG_SKIP_ACTIVATE=1`` forces no source even if ``ICEBERG_SOURCE_ACTIVATE=1``.

**Strict “only under ``github/``”** (no symlinked dataset outside the tree; ``PYTHONPATH`` is only ``github/src`` — set by the Python driver, not inherited):

```bash
export ICEBERG_STRICT_GITHUB=1
bash sh/retrain_ficeberg_split1_newloc.sh
```

If data are symlinked to e.g. ``…/iceberg/data/…``, strict mode exits until you copy or bind-mount under ``github/data/``.

**Other setups:** use a scientific Python (conda/venv), not bare cluster `python3`:

```bash
cd /path/to/iceberg/github
conda activate ms-gen   # or your env name
bash sh/retrain_ficeberg_split1_newloc.sh
```

The wrapper prefers, in order: **`ICEBERG_PYTHON`**, then **`$CONDA_PREFIX/bin/python`** if conda is activated, then **`$VIRTUAL_ENV/bin/python`** after `activate_env.sh`, then **`PYTHON`**, then **`python`** on `PATH`, then **`python3`**. On Compute Canada / similar systems, if you see `ModuleNotFoundError: pandas`, use `activate_env.sh`/conda or set e.g. `export ICEBERG_PYTHON=.../bin/python` before `bash sh/...`.

Optional: `export ICEBERG_GITHUB_ROOT=/abs/path/to/FMS2Lib` to override repo root detection.

## `run_predict_testset_inten.sh`

Calls **`src/foriceberg/ms_pred/dag_pred/predict_smis_trained_inten.py`**. Required env: `GEN_CHECKPOINT`, `INTEN_CHECKPOINT`, `TEST_LABELS`, `SAVE_DIR`.

```bash
cd /path/to/FMS2Lib
export GEN_CHECKPOINT="mode/foriceberg/generation/best.ckpt"
export INTEN_CHECKPOINT="mode/foriceberg/intensity/best.ckpt"
export TEST_LABELS="data/foriceberg/spec_datasets/F_ICEBERG_dataset/testset_split_1_labels.tsv"
export SAVE_DIR="script/foriceberg/results/predictions_testset_run1"
bash sh/foriceberg/run_predict_testset_inten.sh
```

## `run_evaluate_testset_cosine.sh`

Calls **`script/foriceberg/evaluate_test_set.py`** on **`${SAVE_DIR}/preds.hdf5`**. Required env: `SAVE_DIR`, `TEST_LABELS`, `OUTPUT_DIR`, `DATASET_NAME`.

```bash
export OUTPUT_DIR="script/foriceberg/results/evaluation_testset_run1"
export DATASET_NAME="F_ICEBERG_dataset"
bash sh/foriceberg/run_evaluate_testset_cosine.sh
```

Equivalent one-shot: **`python script/foriceberg/run_predict_and_evaluate_testset.py`** (see root ``README.md``).

## `legacy_merge_eval_interactive_20260507_101600_args.yaml`

Archived **predict** args from an interactive run on a full ICEBERG checkout (paths inside are **absolute** to that machine). After uploading only `github/`, **edit paths** in the YAML or pass equivalent CLI flags to:

`python src/ms_pred/dag_pred/predict_smis_trained_inten.py ...`

using **`github/src`** on `PYTHONPATH` and labels/checkpoints under your layout.
