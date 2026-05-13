# `smiles_to_predicted_mgf.py`

Predict **MS/MS** from **SMILES** with **ICEBERG** (generation + intensity checkpoints) and write an **MGF** file.

## Paths (script under `github/script/`)

- **`REPO_ROOT`** is the **`github/`** directory (parent of `script/`).
- **Default checkpoints**: `github/mode/generation/best.ckpt`, `github/mode/intensity/best.ckpt` (override with `--gen-checkpoint` / `--inten-checkpoint`).
- **Predictor subprocess**: `github/src/ms_pred/dag_pred/predict_smis_trained_inten.py`. **`PYTHONPATH`** includes **`github/src`** so `import ms_pred` works.
- **`--out`**: output MGF path; keep paths under `github/` if you want a self-contained tree (e.g. `database/`).

Install the same **Python** stack as ICEBERG (PyTorch, PyTorch Lightning, DGL, torch-scatter, RDKit, h5py, pandas, etc.).

---

## CLI summary

| Argument | Description |
|----------|-------------|
| `--smiles` or `--smiles-file` | One required: single SMILES, or a text file with one SMILES per line |
| `--out` | Output MGF path (required) |
| `--gen-checkpoint` / `--inten-checkpoint` | Optional; defaults above |
| `--gpu` | Optional; passed through to the predictor |
| `--batch-size`, etc. | Optional; defaults 8 / 0 / 100 / 2000 / 0.1 |

---

## Examples (paths only under `github/`)

Set **`GITHUB`** to the absolute path of your **`github`** directory (the folder that contains `script/`, `src/`, `mode/`, `database/`). Do not use angle-bracket placeholders in real commands.

**Single SMILES:**

```bash
GITHUB="/path/to/iceberg/github"

python "${GITHUB}/script/smiles_to_predicted_mgf.py" \
  --smiles "CC(=O)O" \
  --out "${GITHUB}/database/pred_single.mgf"
```

**Read SMILES from a file under `database/`:**

```bash
GITHUB="/path/to/iceberg/github"

python "${GITHUB}/script/smiles_to_predicted_mgf.py" \
  --smiles-file "${GITHUB}/database/example_smiles.txt" \
  --out "${GITHUB}/database/pred_from_file.mgf"
```

Append **`--gpu`** when a GPU is available.

On success the script prints `Wrote: ...` with the absolute path of the output MGF.
