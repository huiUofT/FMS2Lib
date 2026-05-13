# `run_iceberg_retrieval_eval_evaltest.py`

End-to-end **candidate retrieval evaluation** for ICEBERG: read a candidate CSV (e.g. from `build_pubchem_top100_by_ecfp4_from_mgf.py`) and the matching **MGF**, build predictor labels, run `predict_smis_trained_inten.py`, then score candidates with a **binned cosine** metric aligned with `evaluate_test_set`-style `dist_bin` (not the plain vector cosine used in `ms_pred.retrieval.run_iceberg_retrieval_eval`).

Outputs include aggregate Top-1 accuracy and a **per-spectrum** TSV.

## Prerequisites

- Same stack as the ICEBERG repo predictor: PyTorch, DGL, etc., as required by `predict_smis_trained_inten.py`
- **h5py**, **numpy**, **pandas**, **RDKit** (used by `ms_pred.retrieval.run_iceberg_retrieval_eval`)
- Wrappers add **`github/src`** to `sys.path` so **`ms_pred.retrieval`** is importable (see [Python path](#python-path))

## Python path

Implementation lives under **`github/src/ms_pred/retrieval/`** (`run_iceberg_retrieval_eval.py` + `run_iceberg_retrieval_eval_evaltest.py`). The thin wrappers **`github/script/run_iceberg_retrieval_eval_evaltest.py`** and **`github/run_iceberg_retrieval_eval_evaltest.py`** prepend `github/src` to `sys.path` and call `main()`.

If you import from another entry point, ensure **`PYTHONPATH`** includes the `github/src` directory (the parent of the `ms_pred` package), for example:

```bash
export PYTHONPATH="/path/to/FMS2Lib/src/foriceberg:${PYTHONPATH}"
```

The default `--predict-script` inside `ms_pred.retrieval.run_iceberg_retrieval_eval` points at **`github/src/ms_pred/dag_pred/predict_smis_trained_inten.py`**. The predictor process still needs that tree on the interpreter path (the wrappers and `blind_retrieval_from_pred_formula.py` set `PYTHONPATH` for subprocesses where applicable).

## Inputs

### Candidate CSV (`--csv`)

Must be readable by `csv.DictReader` with at least:

- `identifier` — must match the spectrum ID in the MGF (`TITLE`, `IDENTIFIER`, or `FEATURE_ID`)
- `true_smiles` — ground-truth SMILES for Top-1 checks
- `candidate_smiles` — one row per candidate in ranking order (multiple rows per `identifier`)

This matches the CSV written by `build_pubchem_top100_by_ecfp4_from_mgf.py`.

### MGF (`--mgf`)

Used for two purposes:

1. **Metadata** (precursor m/z, instrument, etc.) keyed by spectrum ID, to fill `candidates_labels.tsv` for prediction  
2. **Truth MS/MS** — peak lines inside each block are binned with the same `num_bins` / `upper_limit` as in the predictions HDF5 for scoring  

Keys are normalized to uppercase (e.g. `PEPMASS` / `PRECURSOR_MZ`, `IDENTIFIER`, …). If a spectrum has no peaks in the MGF, it still appears in the per-spectrum table with a note such as `no_experimental_spectrum_in_mgf`.

### Checkpoints

- `--gen-checkpoint` — generation checkpoint (required)  
- `--inten-checkpoint` — intensity checkpoint (required)  

### Other CLI flags

Forwarded through `base.parse_args()` from `ms_pred.retrieval.run_iceberg_retrieval_eval`:

| Argument | Default | Description |
|----------|---------|-------------|
| `--out-dir` | (required) | Run output directory |
| `--predict-script` | `.../src/ms_pred/dag_pred/predict_smis_trained_inten.py` | Override if your layout differs |
| `--batch-size` | `32` | Predictor batch size |
| `--num-workers` | `0` | DataLoader workers |
| `--sparse-k` | `100` | Sparse output top-k |
| `--max-nodes` | `5000` | Graph cap |
| `--threshold` | `0.05` | Intensity threshold |
| `--mz-tol-da` | `0.01` | Used by the base module’s other helpers; evaltest path uses binned cosine |
| `--python` | `sys.executable` | Python for the predictor subprocess |
| `--gpu` | off | Pass `--gpu` to the predictor |

**Collision energy:** labels are written with **`['30.0']`**, and the evaltest scorer uses the **`30`** eV channel when calling `dist_bin`. Your MGF experimental spectra should be comparable to that setting for meaningful scores.

## Usage

From the ICEBERG repository root (adjust paths):

```bash
cd /path/to/FMS2Lib

python github/script/run_iceberg_retrieval_eval_evaltest.py \
  --csv /path/to/pubchem_top100_by_ecfp4.csv \
  --mgf /path/to/queries.mgf \
  --out-dir /path/to/run_out \
  --gen-checkpoint /path/to/gen_best.ckpt \
  --inten-checkpoint /path/to/inten_best.ckpt
```

Add `--gpu` if appropriate.

## Outputs (under `--out-dir`)

| File | Description |
|------|-------------|
| `candidates_labels.tsv` | Tab-separated labels for every `(identifier, candidate index)` row |
| `preds/preds.hdf5` | ICEBERG predictions |
| `top1_summary.csv` | `top1_correct`, `total`, `top1_acc_pct` (canonical SMILES match at rank 1) |
| `per_spectrum_retrieval.tsv` | One row per spectrum: Top-1 hit flag, rank of true SMILES among candidates, similarities, notes |

## Relation to `ms_pred.retrieval.run_iceberg_retrieval_eval`

- Same pipeline: CSV + MGF → labels → predictor → Top-1 summary.  
- **Difference:** the evaltest module uses **weighted binned cosine distance** (`dist_bin` with `func="cos"`, sparse expanded to dense bins) consistent with eval-test style scoring, and writes **`per_spectrum_retrieval.tsv`**.

## Upstream workflow

1. Build `--csv` with `build_pubchem_top100_by_ecfp4_from_mgf.py` (`README_build_pubchem_top100_by_ecfp4_from_mgf.md`).  
2. Run this script with the **same** MGF and that CSV.
