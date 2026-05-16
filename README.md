# FMS2Lib

MS2 prediction for fluorinated compounds (formula / ranker workflows in this export).


## Layout

| Path | Contents |
|------|----------|
| `src/msbuddy/` | MSBuddy (third-party package only) |
| `script/formula_prediction/calibration/` | MGF calibration (`clean.py`, `subformula.py`, …) |
| `script/formula_prediction/netural_loss/` | NL pipeline (sibling scripts; subprocess paths) |
| `script/formula_prediction/train/` | Feature build + LightGBM ranker training |
| `script/formula_prediction/eval/` | Eval joblib generation + inference |
| `data/testdataset/` | Small bundled MGFs, `test10_formulas.txt`, **`test10_nl_meta.csv`** (FEATURE_ID + SMILES + FORMULA for NL step 3) |

## Paths

- **`GITHUB_ROOT`**: absolute path to **this** `github` directory (export root).
- **`--msbuddy-dir`**: directory that **contains** the `msbuddy/` package → **`"${GITHUB_ROOT}/src"`**.
- **`PYTHONPATH`**: optional if scripts already discover `src/msbuddy` by walking parents; otherwise **`"${GITHUB_ROOT}/src"`**.
- **Self-contained runs (no paths outside `GITHUB_ROOT`)**: pass **`--repo-root "${GITHUB_ROOT}"`** to `mgf_to_results.py`, and use **absolute paths** for every file argument (`--mgf`, `--formulas`, `--output`, `--output-dir`, `--model`, …) under **`${GITHUB_ROOT}/...`**. Put scratch outputs under e.g. **`${GITHUB_ROOT}/scratch/...`**.
- **Still outside this folder (unavoidable)**: the Python interpreter, installed packages in your environment, and any **on-demand downloads** triggered by `msbuddy` (caches not stored in `github/` unless you configure them).

## Example commands (bash), self-contained

Set once per shell (replace `/path/to/f-paper/github` with your real absolute path):

```bash
export GITHUB_ROOT=/path/to/f-paper/github
cd "$GITHUB_ROOT"
export PYTHONPATH="${GITHUB_ROOT}/src"
export PYTHONIOENCODING=utf-8
mkdir -p "${GITHUB_ROOT}/scratch"
```

**MGF calibration** (`clean.py`): MS1/MS2 tolerances are in ppm. `--ms2-tol` (default 5) drives MS2 preprocessing and `assign_subformula`; `--ms1-tol` (default 2) is recorded for consistency.

```bash
python "${GITHUB_ROOT}/script/formula_prediction/calibration/clean.py" \
  --mgf "${GITHUB_ROOT}/data/testdataset/F_ICEBERG_calibrated_test10.mgf" \
  --formulas "${GITHUB_ROOT}/data/testdataset/test10_formulas.txt" \
  --output "${GITHUB_ROOT}/scratch/calibrated_out.mgf" \
  --ms1-tol 2 \
  --ms2-tol 5 \
  --msbuddy-dir "${GITHUB_ROOT}/src"
```

**Neutral-loss pipeline** (`nl_analysis_pipeline.py`): forwards `--ms1-tol` / `--ms2-tol` to `subformula.py` (same defaults as above).

Step 3 (NL frequency by compound) needs a metadata CSV with **`FEATURE_ID`** and the compound column (**`SMILES`** by default, or set `--compound-key`). If `--formulas` is a **`.txt`**, either:

- pass **`--meta-csv`** (example bundle: `data/testdataset/test10_nl_meta.csv` — aligned with `F_ICEBERG_calibrated_test10.mgf`), or  
- rely on **auto meta from MGF**: when each `BEGIN IONS` block includes `FEATURE_ID=` and `SMILES=` lines, the pipeline writes `{mgf_stem}_nl_meta_from_mgf.csv` under `--output-dir`.

```bash
python "${GITHUB_ROOT}/script/formula_prediction/netural_loss/nl_analysis_pipeline.py" \
  --mgf "${GITHUB_ROOT}/data/testdataset/F_ICEBERG_calibrated_test10.mgf" \
  --formulas "${GITHUB_ROOT}/data/testdataset/test10_formulas.txt" \
  --meta-csv "${GITHUB_ROOT}/data/testdataset/test10_nl_meta.csv" \
  --output-dir "${GITHUB_ROOT}/scratch/nl_out" \
  --ms1-tol 2 \
  --ms2-tol 5 \
  --msbuddy-dir "${GITHUB_ROOT}/src"
```

**Train (features + ranker)** (`mgf_to_results.py`): supervised only — **`--formulas`** must be a **`.txt`** with one precursor formula per line, **same order as spectra in the MGF**. No compound metadata CSV is used here (that CSV is only for the NL pipeline step 3).

```bash
python "${GITHUB_ROOT}/script/formula_prediction/train/mgf_to_results.py" \
  --mgf "${GITHUB_ROOT}/data/testdataset/F_ICEBERG_calibrated_test10.mgf" \
  --formulas "${GITHUB_ROOT}/data/testdataset/test10_formulas.txt" \
  --output-dir "${GITHUB_ROOT}/scratch/train_out" \
  --repo-root "${GITHUB_ROOT}"
```

**Eval (generate + score)** (`run_eval_generate_and_predict.py`; adjust `--model` to a joblib under `${GITHUB_ROOT}`):

```bash
python "${GITHUB_ROOT}/script/formula_prediction/eval/run_eval_generate_and_predict.py" \
  --mgf "${GITHUB_ROOT}/data/testdataset/F_ICEBERG_calibrated_test10.mgf" --mgf-only \
  --output-dir "${GITHUB_ROOT}/scratch/eval_out" \
  --model "${GITHUB_ROOT}/scratch/train_out/trained_models/model_ms1_ms2_ranker.joblib" \
  --no-mmap
```

PowerShell (environment only):

```powershell
$env:GITHUB_ROOT = "D:\path\to\f-paper\github"
$env:PYTHONPATH = "$env:GITHUB_ROOT\src"
$env:PYTHONIOENCODING = "utf-8"
```

For all flags, run `python <script.py> --help`.

## Dependencies

One file covers **MSBuddy / formula_prediction** and **ICEBERG foriceberg** (see comments in the file). For ICEBERG, install CUDA-matched **torch / dgl / PyG** first when you need GPU training.

```bash
pip install -r requirements.txt
pip install -e .   # registers ``msbuddy`` from ``src/`` (editable)
```

**MsBuddy data (`src/msbuddy/data/*.joblib`):** these files are **not** stored in the repo (GitHub file-size limit). On first use, `msbuddy` downloads `common_db_v0.2.4.joblib` and `formula_db_v0.2.4.joblib` from the upstream release URLs in `src/msbuddy/load.py` (~416 MB for the formula DB). Ensure network access for the first run.

Longer docs (Chinese/English) and per-folder READMEs remain under `f-paper/py文件/` in the main repository.

---

## ICEBERG / `foriceberg` (MS/MS prediction & retrain)

Portable ICEBERG workflows live under **`*/foriceberg/`** subfolders (not a separate top-level tree). Code: **`src/foriceberg/ms_pred`**. Scripts: **`script/foriceberg/`**. Data: **`data/foriceberg/spec_datasets/`**. Checkpoints: **`mode/foriceberg/`**. Shell: **`sh/foriceberg/`**.

### Layout

```text
FMS2Lib/
  script/foriceberg/     # run.py, drivers, results/ (created at train time)
  src/foriceberg/ms_pred/
  data/foriceberg/spec_datasets/<dataset>/
  mode/foriceberg/generation|intensity/best.ckpt
  sh/foriceberg/retrain_ficeberg_split1_newloc.sh
  sh/foriceberg/run_predict_testset_inten.sh
  sh/foriceberg/run_evaluate_testset_cosine.sh
```

### Environment

```bash
export FMS2LIB=/path/to/FMS2Lib
cd "${FMS2LIB}"
export PYTHONPATH="${FMS2LIB}/src/foriceberg:${PYTHONPATH:-}"
# Optional override:
# export ICEBERG_GITHUB_ROOT="${FMS2LIB}"
```

**Dependencies:** same root **`requirements.txt`** (ICEBERG section) or parent ICEBERG `activate_env.sh`.

### Data (`data/foriceberg/spec_datasets/F_ICEBERG_dataset/`)

Needs: `labels.tsv`, `spec_files.hdf5`, `splits/split_1.tsv`, `magma_outputs/magma_tree.hdf5`, `subformulae/no_subform.hdf5` (symlink large data if needed):

```bash
mkdir -p "${FMS2LIB}/data/foriceberg/spec_datasets"
ln -sfn /path/to/iceberg/data/spec_datasets/F_ICEBERG_dataset \
  "${FMS2LIB}/data/foriceberg/spec_datasets/F_ICEBERG_dataset"
```

### Unified CLI

```bash
cd "${FMS2LIB}"
python script/foriceberg/run.py                  # list commands
python script/foriceberg/run.py <cmd> --help
```

| Command | Purpose |
|---------|---------|
| `process-dataset` | MGF → `data/foriceberg/spec_datasets/<name>/` |
| `magma` | → `magma_outputs/magma_tree.hdf5` |
| `retrain` | F_ICEBERG split_1 retrain (or `bash sh/foriceberg/retrain_ficeberg_split1_newloc.sh`) |
| `predict-eval` | test labels → preds + cosine eval (or two-step: `sh/foriceberg/run_predict_testset_inten.sh` + `run_evaluate_testset_cosine.sh`) |
| `smiles2mgf` / `end2end` / `retrieval-eval` / `blind-retrieval` / `pubchem-top100` | see `python script/foriceberg/run.py` |

### Typical pipeline

1. `python script/foriceberg/run.py process-dataset --mgf-file ... --dataset-name ...`
2. `python script/foriceberg/run.py magma --spectra-dir data/foriceberg/spec_datasets/.../spec_files.hdf5 ...`
3. `bash sh/foriceberg/retrain_ficeberg_split1_newloc.sh`
4. `python script/foriceberg/run.py predict-eval --dataset-labels data/foriceberg/spec_datasets/.../testset_split_1_labels.tsv ...`

Example **predict-eval** (paths relative to **`FMS2LIB`** root):

```bash
python script/foriceberg/run.py predict-eval \
  --dataset-labels data/foriceberg/spec_datasets/F_ICEBERG_dataset/testset_split_1_labels.tsv \
  --save-dir script/foriceberg/results/predictions_testset_run1 \
  --output-dir script/foriceberg/results/evaluation_testset_run1 \
  --dataset-name F_ICEBERG_dataset \
  --gpu
```

Retrain outputs go under **`script/foriceberg/results/`** by default.
