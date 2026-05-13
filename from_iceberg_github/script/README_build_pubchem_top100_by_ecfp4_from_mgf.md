# `build_pubchem_top100_by_ecfp4_from_mgf.py`

Build a ranked candidate list per spectrum: for each query in an MGF, take PubChem structures that share the same molecular formula, rank them by **ECFP4 Tanimoto similarity** (Morgan fingerprint, radius 2, 2048 bits) to the query SMILES, and write a CSV suitable for downstream MS/MS retrieval evaluation.

## Prerequisites

- Python 3 with **RDKit**, **h5py**, **pandas**, **tqdm**

## Inputs

### MGF (`--mgf`)

Parsed as standard MGF blocks (`BEGIN IONS` … `END IONS`). For each block the script needs:

| Role | MGF fields (first match wins where noted) |
|------|-------------------------------------------|
| Spectrum ID | `TITLE`, `IDENTIFIER`, or `FEATURE_ID` |
| Ground-truth structure | `SMILES` |
| Formula (optional) | `FORMULA`; if absent, the formula is computed from `SMILES` with RDKit |

If ID or SMILES is missing, or no formula can be obtained, that block contributes no rows.

### PubChem HDF5 (`--pubchem-hdf5`)

An HDF5 file whose groups are keyed by **molecular formula** (normalized string preferred; the script also tries the raw formula string from the MGF). Under each formula group, candidate SMILES are collected recursively from datasets whose names are `smiles`, `smiles_list`, or `smiles_str` (case-insensitive). InChI-like strings are skipped.

If a formula has no matching group or no SMILES datasets, that spectrum produces no output rows.

## Usage

```bash
python build_pubchem_top100_by_ecfp4_from_mgf.py \
  --mgf /path/to/queries.mgf \
  --pubchem-hdf5 /path/to/pubchem_by_formula.h5 \
  --out-csv /path/to/candidates.csv
```

### Optional arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--max-topk` | `100` | Maximum number of PubChem-ranked candidates per spectrum (after the query itself). |
| `--workers` | `1` | If `> 1`, process MGF blocks in parallel chunks (each worker opens its own HDF5 read handle). |
| `--chunk-size` | `200` | Number of MGF blocks per worker task when `--workers > 1`. |

## Output CSV

Columns:

- `identifier` — spectrum ID from the MGF  
- `true_formula` — normalized formula used for HDF5 lookup  
- `true_smiles` — query SMILES from the MGF  
- `candidate_smiles` — query SMILES (rank 1, Tanimoto 1.0) followed by PubChem candidates in descending Tanimoto order, canonical-SMILES deduplicated  
- `tanimoto` — ECFP4 Tanimoto to the query (1.0 for the query itself)  
- `rank` — 1-based rank within that spectrum’s candidate list  

## Next step

Feed `--out-csv` together with the **same** MGF into `run_iceberg_retrieval_eval_evaltest.py` (see `README_run_iceberg_retrieval_eval_evaltest.md`) to run ICEBERG predictions and spectrum-based Top-1 retrieval evaluation.
