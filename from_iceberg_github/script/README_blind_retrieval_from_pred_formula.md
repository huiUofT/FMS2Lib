# `blind_retrieval_from_pred_formula.py`

Blind candidate retrieval: use a **predicted molecular formula per spectrum** (from a CSV), gather SMILES from **SusDat** and/or **PubChem-by-formula HDF5**, run **ICEBERG** (generation + intensity checkpoints) on every candidate, and rank by **binned cosine similarity** to the **experimental MS/MS** in an MGF.

**Important:** `SMILES=` in the MGF header is **not** used to build or score candidates. Only peak lists and precursor-related metadata from the MGF are used for prediction inputs and scoring. Optional `--audit-mgf-smiles` adds a read-only column for offline checks.

## Layout assumption

The script lives under `iceberg/github/script/` and prepends **`github/src`** to `sys.path` so it can import **`ms_pred.retrieval.run_iceberg_retrieval_eval`** (MGF IO, labels, predictor wiring) and **`dist_bin`** from **`ms_pred.retrieval.run_iceberg_retrieval_eval_evaltest`**. Those modules live under `github/src/ms_pred/retrieval/` (no dependency on `iceberg/retraiveal/` for this script).

`build_pubchem_top100_by_ecfp4_from_mgf.py` remains alongside this script in `github/script/` for formula normalization and PubChem HDF5 reading.

Default checkpoints and predictor live under **`github/`**:

| Item | Default path |
|------|----------------|
| Gen checkpoint | `github/mode/generation/best.ckpt` |
| Inten checkpoint | `github/mode/intensity/best.ckpt` |
| Predict script | `github/src/ms_pred/dag_pred/predict_smis_trained_inten.py` |

The predictor subprocess runs with `cwd` = `github/` and `PYTHONPATH` including `github/src`.

## Dependencies

- Python: **RDKit**, **h5py**, **numpy**, **pandas**
- Same PyTorch / DGL stack as `predict_smis_trained_inten.py`
- `build_pubchem_top100_by_ecfp4_from_mgf.py` in the same `script/` directory (for formula normalization and PubChem HDF5 reading)

## Inputs

### `--mgf`

Experimental MGF. Spectrum IDs must align with the prediction CSV (`TITLE`, `IDENTIFIER`, or `FEATURE_ID`, same resolution as `read_mgf_metadata_and_truth` in `ms_pred.retrieval.run_iceberg_retrieval_eval`). Peak lines are binned using HDF5 attrs from the prediction output (`num_bins`, `upper_limit`).

### `--pred-csv`

Required columns:

- **`identifier`** — matches the MGF spectrum ID.
- **Formula column** — one of `pred_formula`, `predicted_formula`, or `formula` (case-insensitive header match).

Optional:

- **`precursor_mz`** (or `precursor mz`) — used only as fallback text in outputs if MGF precursor is missing.

Candidates are keyed by **normalized predicted formula only**. The `FORMULA=` line in the MGF is **not** used for candidate lookup (blind to the true composition in the reference file).

### `--susdat`

CSV with at least **`formula`** (or `Molecular_Formula`), **`SMILES`**, and optionally **`Name`**. Rows whose normalized formula equals the predicted formula contribute candidates.

### `--pubchem-hdf5`

HDF5 with the same convention as `README_build_pubchem_top100_by_ecfp4_from_mgf.md` (formula groups, SMILES in datasets named like `smiles` / `smiles_list` / `smiles_str`). In this repo the canonical file is often `data/pubchem/pubchem_formulae_inchikey.hdf5`; a hard link or copy named `github/database/pubchem_by_formula.h5` is fine.

At least one of `--susdat` or `--pubchem-hdf5` must be given.

## Scoring

Per candidate, predicted MS/MS at **30 eV** is compared to the experimental spectrum using **`dist_bin`** from `ms_pred.retrieval.run_iceberg_retrieval_eval_evaltest` (sparse predictions expanded to dense bins, cosine-based distance, same weighting as eval-test style). Reported **similarity = 1 − distance** (higher is better).

## Outputs (under `--out-dir`)

| File | Description |
|------|-------------|
| `blind_retrieval_top1.csv` | One row per spectrum with best SMILES, `candidate_source`, `pred_formula`, similarity, etc. |
| `skipped_spectra.csv` | Spectra skipped (unknown ID, no candidates, …). |
| `candidates_long.csv` | Long table of all candidates with source and formula metadata. |
| `candidates_for_predictor.csv` | Minimal CSV for label building. |
| `candidates_labels.tsv` | ICEBERG predictor input. |
| `preds/preds.hdf5` | Model output (unless `--skip-predict`). |
| `per_spectrum_all_candidates.tsv` | Only if `--write-ranking-tsv`: all candidates and ranks. |

### `candidate_source` values

`susdat`, `pubchem`, or `susdat+pubchem` when the same canonical SMILES appeared in both libraries.

## Example

From the ICEBERG repo root:

```bash
cd /path/to/FMS2Lib

python github/script/blind_retrieval_from_pred_formula.py \
  --mgf github/database/F_ICEBERG_calibrated_test10.mgf \
  --pred-csv github/database/predictions_model_ms1_ms2_ranker.csv \
  --susdat github/database/susdat_F.csv \
  --pubchem-hdf5 github/database/pubchem_by_formula.h5 \
  --out-dir github/database/blind_retrieval_run1 \
  --write-ranking-tsv \
  --gpu
```

Labels only (no GPU / no prediction):

```bash
python github/script/blind_retrieval_from_pred_formula.py \
  --mgf github/database/F_ICEBERG_calibrated_test10.mgf \
  --pred-csv github/database/predictions_model_ms1_ms2_ranker.csv \
  --susdat github/database/susdat_F.csv \
  --out-dir github/database/blind_retrieval_labels_debug \
  --skip-predict
```

## Useful flags

| Flag | Meaning |
|------|---------|
| `--max-candidates N` | Cap merged candidates per spectrum after deduplication (default `400`; `0` = no cap). SusDat order is preserved first, then PubChem. |
| `--audit-mgf-smiles` | Append `mgf_smiles_audit` from the MGF (does not affect ranking). |
| `--gen-checkpoint` / `--inten-checkpoint` / `--predict-script` | Override defaults. |

## Related docs

- `README_build_pubchem_top100_by_ecfp4_from_mgf.md` — PubChem HDF5 layout and ECFP4 candidate tables (uses true SMILES from the MGF; different use case).
- `README_run_iceberg_retrieval_eval_evaltest.md` — same `dist_bin` scoring on a known truth CSV.
