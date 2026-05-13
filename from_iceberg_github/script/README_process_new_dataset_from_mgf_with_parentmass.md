# `process_new_dataset_from_mgf_with_parentmass.py`

Build a full **`spec_datasets/<name>/`** layout from an **MGF** file: **`labels.tsv`**, **`spec_files.hdf5`**, and a **stratified train/val/test split** by fluorine count in the SMILES string.

Compared to a plain `process_new_dataset_from_mgf`-style flow, this script writes an extra **`#parentmass`** line into each spectrum’s **SIRIUS-style** text stored in HDF5. The parent mass is set **equal to the numeric value on the `#PRECURSORMZ` line** (same convention as the official **`F_ICEBERG_dataset`** `spec_files` payload: `#parentmass` immediately follows `#PRECURSORMZ`).

## Prerequisites

- **RDKit**, **pandas**, **numpy**, **tqdm**
- **`ms_pred.common`** (for `HDF5Dataset`): add the tree that contains the `ms_pred` package to **`PYTHONPATH`**. For the self-contained **`github/`** tree:

```bash
export GITHUB="/path/to/iceberg/github"
export PYTHONPATH="${GITHUB}/src:${PYTHONPATH:-}"
```

## Working directory

Output paths are **relative to the current working directory**:

`data/spec_datasets/<dataset-name>/`

Run the script from the project root where you want that `data/` directory to appear (often the **ICEBERG repo root** or the **`github/`** root if you keep datasets under `github/data/`).

## MGF fields

Each `BEGIN IONS` … `END IONS` block must include at least:

- `IDENTIFIER=`
- `SMILES=`
- `PRECURSOR_MZ=`
- One or more peak lines: `m/z intensity`

Optional headers (used when present):

- `CHARGE=`, `ADDUCT=`, `INSTRUMENT_TYPE=`, `FEATURE_ID=`, `COLLISION_ENERGY=` or `CE=`

Blocks without `spec` + `smiles` or without peaks are skipped.

## Outputs

Under `data/spec_datasets/<dataset-name>/`:

| Artifact | Description |
|----------|-------------|
| `labels.tsv` | `spec`, `smiles`, `inchikey`, `ionization`, `precursor`, `instrument`, `collision_energies` |
| `spec_files.hdf5` | One string per spectrum: `#` metadata (including **`#parentmass`**), blank line, `>collision …`, peak list |
| `splits/split_<seed>.tsv` | Columns `spec`, `Fold_0` (`train` / `val` / `test`) |

Stratification: spectra are grouped by **count of `"F"` in the SMILES string** (proxy for fluorination level), then split with the requested fractions inside each group.

## Usage

```bash
cd /path/to/project/root   # parent of data/spec_datasets
export PYTHONPATH="/path/to/iceberg/github/src:${PYTHONPATH:-}"

python /path/to/iceberg/github/script/process_new_dataset_from_mgf_with_parentmass.py \
  --mgf-file /path/to/input.mgf \
  --dataset-name F_ICEBERG_dataset_repro_with_pm \
  --train-frac 0.8 --val-frac 0.1 --test-frac 0.1 \
  --seed 1
```

## Next steps (pipeline)

After this script, the usual F_ICEBERG-style pipeline continues with **MAGMA** trees, **generation** training, **intensity** training, etc., as in the upstream project docs.

## Legacy path

`github/process_new_dataset_from_mgf_with_parentmass.py` may remain as a thin launcher that forwards to **`github/script/process_new_dataset_from_mgf_with_parentmass.py`** so old command lines keep working.
