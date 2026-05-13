# `run_end2end_from_mgf_susdat_onepass.py`

One-pass pipeline: **MGF spectra + SusDat-style compound table** → build per-spectrum candidate lists by molecular formula → run the end-to-end prediction script (cosine ranking) → evaluate Top-1 retrieval using **RDKit canonical SMILES**.

Upload this **`github/`** tree (not only `script/`): entry scripts live in **`github/script/`**, and the bundled **`ms_pred`** package lives in **`github/src/ms_pred/`** (same layout as upstream ICEBERG `src/ms_pred`). Both paths are wired via `sys.path` to `github/src/`.

Requires a **Python** installation with **PyTorch**, **PyTorch Lightning**, **DGL**, **torch-scatter**, **pygmtools**, **RDKit**, **pandas**, **numpy**, **tqdm**, **PyYAML**, and **h5py** (and any other imports your checkpoints require). You do **not** need the rest of the ICEBERG repository on disk for imports, but you still need compatible **checkpoint** files and GPU drivers if you pass `--gpu` downstream.

---

## Inputs

### SusDat CSV

Required columns:

- **`formula`** — molecular formula as a string (or **`Molecular_Formula`**, which is renamed to `formula`).
- **`Name`** — compound name; used to attach a ground-truth SMILES to each spectrum when the MGF does not carry `SMILES=`.
- **`SMILES`** — structures indexed by formula and name.

The script normalizes formulas (Hill-style ordering) and groups all SMILES sharing the same normalized formula as retrieval candidates.

### MGF

Each spectrum must be a standard block:

```text
BEGIN IONS
TITLE=...
FORMULA=...
SMILES=...          # optional; ground truth if present
... other headers ...
<m/z> <intensity>   # peak lines: first character is a digit, no '=' in the line
END IONS
```

**Requirements:**

- **`FORMULA=`** — mandatory for inclusion. Spectra without a usable formula are skipped.
- **`TITLE=`** — used as the spectrum **`identifier`** in downstream CSVs. If **`SMILES=`** is missing, the script matches **`TITLE`** to **`Name`** in the SusDat table (exact match, then case-insensitive match on stripped names) to obtain the ground-truth SMILES.
- **Candidates** — the normalized MGF formula must appear in the SusDat-derived formula→SMILES index; otherwise the spectrum is skipped (no candidates).
- **Peak lines** — lines that look like `m/z intensity` (line starts with a digit) are treated as peaks; header lines use `KEY=value`.

---

## Command-line interface

| Argument | Required | Description |
|----------|----------|-------------|
| `--mgf` | yes | Input MGF path |
| `--susdat` | yes | SusDat CSV path |
| `--gen-checkpoint` | no | Default: **`github/mode/generation/best.ckpt`** (next to `script/`) |
| `--inten-checkpoint` | no | Default: **`github/mode/intensity/best.ckpt`** |
| `--out-dir` | yes | Output directory (created if missing) |
| `--predict-script` | no | Default: **`src/ms_pred/dag_pred/predict_eval_end2end_pubchem_candidates_cosine.py`** under the same `github/` tree |
| `--batch-size` | no | default `32` |
| `--num-workers` | no | default `0` |
| `--sparse-k` | no | default `100` |
| `--max-nodes` | no | default `5000` |
| `--threshold` | no | default `0.05` |

If you **omit** all of `--batch-size`, `--num-workers`, `--sparse-k`, `--max-nodes`, and `--threshold`, the subprocess receives exactly these defaults (they are forwarded to the prediction driver).

The script runs the predictor as a subprocess, then reads its outputs under `--out-dir/pred_run/`.

---

## Outputs (under `--out-dir`)

| Path | Purpose |
|------|---------|
| `build_stats.json` | Counts: total blocks, kept with candidates, skipped reasons |
| `candidates_compat_by_formula_from_mgf.csv` | Per-spectrum rows: `identifier`, `SMILES` (truth), `pubchem_smiles_*` columns |
| `targets_with_candidates.mgf` | MGF subset that had candidates |
| `per_entry_candidate_counts.csv` | `identifier`, `SMILES`, `num_candidates` |
| `pred_run/` | Predictor working directory (checkpoints passed through) |
| `pred_run/end2end_pubchem_candidates_top1.csv` | Expected Top-1 file from the predictor |
| `pred_run/retrieval_eval_standardized_summary.json` | Summary metrics after canonical evaluation |
| `pred_run/eval_with_canonical_smiles.csv` | Per-spectrum merged evaluation |

---

## Example (paths hidden; `github/` tree only)

Assume **`GITHUB`** is the top-level folder that contains **`script/`**, **`src/`**, **`mode/`**, and (for this copy-paste) **`database/`**. Replace `<path-to-github>` with that directory on your machine. No username or home directory appears in the command.

**What runs from `github/` only**

- **Code:** `script/run_end2end_from_mgf_susdat_onepass.py` (entry only), **`src/ms_pred/dag_pred/predict_eval_end2end_pubchem_candidates_cosine.py`**, **`src/ms_pred/dag_pred/predict_smis_trained_inten.py`**, and the rest of **`src/ms_pred/`** (import root `github/src/`).
- **Weights (defaults):** `mode/generation/best.ckpt`, `mode/intensity/best.ckpt`.
- **Sample inputs (optional):** `database/mini_onepass_test.mgf` and `database/susdat_F.csv` if you keep them next to the scripts.
- **Output:** any writable directory; the example below writes under `database/onepass_demo_out`.

You still need a **Python** interpreter with the **PyTorch / DGL / RDKit / …** stack installed (see top of this file). That interpreter is not inside `github/`; everything **imported and read as project files** for this pipeline is under **`GITHUB`**.

```bash
GITHUB="<path-to-github>"

OUT="${GITHUB}/database/onepass_demo_out"
rm -rf "${OUT}"
mkdir -p "${OUT}"

python "${GITHUB}/script/run_end2end_from_mgf_susdat_onepass.py" \
  --mgf "${GITHUB}/database/mini_onepass_test.mgf" \
  --susdat "${GITHUB}/database/susdat_F.csv" \
  --out-dir "${OUT}"
```

Default checkpoints resolve to **`${GITHUB}/mode/generation/best.ckpt`** and **`${GITHUB}/mode/intensity/best.ckpt`**. Use your own MGF, SusDat CSV, and `--out-dir` by changing the three arguments above; add `--gen-checkpoint` / `--inten-checkpoint` only if your checkpoints live elsewhere.

Optional: `--predict-script` overrides the bundled driver (default: `src/ms_pred/dag_pred/predict_eval_end2end_pubchem_candidates_cosine.py`).

---

## Notes

- **Ground truth in candidates** — the script inserts the truth SMILES at the front of the candidate list when known, then deduplicates; evaluation and coverage statistics reflect the lists actually used when `candidates_index.csv` is present in `pred_run/`.
- **Evaluation** — post hoc metrics use **RDKit** `MolFromSmiles` / `MolToSmiles(..., canonical=True)`; this may differ from other canonicalization used elsewhere in the project.
