#!/usr/bin/env python3
"""This code was obtained from the ICEBERG implementation developed by
Goldman et al.

Original repository:
https://github.com/samgoldman97/ms-pred

No modifications were made to the original code.

The original copyright notice and license terms are retained.

Score candidate SMILES for each query spectrum by ICEBERG predicted-spectrum cosine similarity.

Inputs:
  - Retrieval CSV containing: identifier, smiles, pubchem_smiles_1..pubchem_smiles_10 (and other cols)
  - Target MGF containing spectra with TITLE matching `identifier`

Method:
  - For each identifier, build up to 11 candidate SMILES (smiles + pubchem_smiles_1..10)
  - Run ICEBERG joint prediction (GEN + INTEN) on all candidates (without modifying official code)
  - Parse the target MGF, bin true spectra, and compute cosine similarity per candidate
  - Pick the highest-cosine candidate as the predicted structure
  - Report top-1 accuracy where `smiles` column is treated as the ground truth

This script calls `predict_smis_trained_inten.py` in the same directory via subprocess.

When shipped under `github/`, default checkpoints resolve to `github/mode/...` (see argparse defaults).
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# `ms_pred` package root is `.../src`; this file lives in `.../src/ms_pred/dag_pred/`.
_SCRIPT_DIR = Path(__file__).resolve().parent
_GITHUB_SRC = _SCRIPT_DIR.parent.parent  # .../src
if str(_GITHUB_SRC) not in sys.path:
    sys.path.insert(0, str(_GITHUB_SRC))

_GITHUB_TOP = Path(__file__).resolve().parents[3]  # parent of `src/` = `github/`
_DEFAULT_GEN_CKPT = _GITHUB_TOP / "mode" / "generation" / "best.ckpt"
_DEFAULT_INTEN_CKPT = _GITHUB_TOP / "mode" / "intensity" / "best.ckpt"

import ms_pred.common as common  # noqa: E402

import re  # noqa: E402


NUMERIC_PAIR_RE = re.compile(
    r"^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$"
)


def _safe_strip(x: object) -> str:
    if x is None:
        return ""
    s = str(x)
    if s.lower() == "nan":
        return ""
    return s.strip()


def _iter_candidate_smiles(row: pd.Series) -> List[str]:
    """
    Build candidate list for a row:
    - Always include 'smiles' if present
    - Then include ALL columns whose names start with 'pubchem_smiles_' (no upper bound),
      ordered by numeric suffix if present, otherwise lexicographically.
    - Deduplicate while preserving order.
    """
    cands: List[str] = []
    # 1) true smiles
    if "smiles" in row:
        s = _safe_strip(row["smiles"])
        if s:
            cands.append(s)
    # 2) dynamic pubchem_smiles_* columns
    pc_cols = [c for c in row.index if isinstance(c, str) and c.startswith("pubchem_smiles_")]
    def _key(col: str):
        try:
            return int(col.split("_")[-1])
        except Exception:
            return col
    pc_cols.sort(key=_key)
    for c in pc_cols:
        s = _safe_strip(row[c])
        if s and s not in cands:
            cands.append(s)
    return cands


@dataclass(frozen=True)
class MGFEntry:
    title: str
    pepmass: float
    peaks: np.ndarray  # shape [N, 2] (mz, intensity)


def parse_mgf_titles(path: Path) -> Dict[str, MGFEntry]:
    """Minimal MGF parser: extracts TITLE, PEPMASS, and peaks."""
    entries: Dict[str, MGFEntry] = {}
    cur_meta: Dict[str, str] = {}
    peaks: List[Tuple[float, float]] = []
    in_block = False

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line == "BEGIN IONS":
                in_block = True
                cur_meta = {}
                peaks = []
                continue
            if line == "END IONS":
                if in_block and "TITLE" in cur_meta and "PEPMASS" in cur_meta and peaks:
                    title = cur_meta["TITLE"].strip()
                    try:
                        pepmass = float(str(cur_meta["PEPMASS"]).split()[0])
                    except Exception:
                        pepmass = float("nan")
                    if not math.isnan(pepmass):
                        arr = np.array(peaks, dtype=np.float32)
                        entries[title] = MGFEntry(title=title, pepmass=pepmass, peaks=arr)
                in_block = False
                continue

            if not in_block:
                continue

            if "=" in line and not line[0].isdigit():
                k, v = line.split("=", 1)
                cur_meta[k.strip()] = v.strip()
                continue

            m = NUMERIC_PAIR_RE.match(line)
            if m:
                peaks.append((float(m.group(1)), float(m.group(2))))

    return entries


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-22
    return float(np.dot(a, b) / denom)


def load_pred_binned_from_preds_hdf5(preds_hdf5: Path, spec_name: str) -> Dict[str, np.ndarray]:
    """
    Load a predicted spectrum for one spec from preds.hdf5 produced by predict_smis_trained_inten.py.
    Returns: dict collision_energy_key -> dense binned vector (np.ndarray)
    """
    import h5py  # local import

    out: Dict[str, np.ndarray] = {}
    pred_key = f"pred_{spec_name}"
    with h5py.File(preds_hdf5, "r") as f:
        if pred_key not in f:
            return out
        pred_group = f[pred_key]
        sparse_out = bool(f.attrs.get("sparse_out", True))
        num_bins = int(f.attrs.get("num_bins", 15000))
        upper_limit = float(f.attrs.get("upper_limit", 1500))

        for ikey in pred_group.keys():
            ikey_group = pred_group[ikey]
            for ce_k in ikey_group.keys():
                ce_group = ikey_group[ce_k]
                if "spec" not in ce_group:
                    continue
                spec_data = ce_group["spec"][:]
                ce_raw = str(ce_k).replace("collision ", "").strip()
                try:
                    ce_float = float(ce_raw)
                except Exception:
                    ce_float = 30.0
                ce_key = f"{ce_float:.0f}"

                if sparse_out and spec_data.ndim == 2 and spec_data.shape[1] >= 2:
                    binned = common.bin_spectra([spec_data], num_bins=num_bins, upper_limit=upper_limit)[0]
                    out[ce_key] = binned.astype(np.float32, copy=False)
                elif spec_data.ndim == 1:
                    out[ce_key] = spec_data.astype(np.float32, copy=False)
                else:
                    binned = common.bin_spectra([spec_data], num_bins=num_bins, upper_limit=upper_limit)[0]
                    out[ce_key] = binned.astype(np.float32, copy=False)
            break  # usually only one ikey
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", required=True, type=str)
    p.add_argument("--target_mgf", required=True, type=str)
    p.add_argument(
        "--gen_checkpoint",
        default=str(_DEFAULT_GEN_CKPT),
        type=str,
        help="GEN model checkpoint path.",
    )
    p.add_argument(
        "--inten_checkpoint",
        default=str(_DEFAULT_INTEN_CKPT),
        type=str,
        help="INTEN model checkpoint path.",
    )
    p.add_argument("--out_dir", required=True, type=str)

    p.add_argument("--num_bins", default=15000, type=int)
    p.add_argument("--upper_limit", default=1500.0, type=float)
    p.add_argument("--batch_size", default=32, type=int)
    p.add_argument("--num_workers", default=0, type=int)
    p.add_argument("--sparse_k", default=100, type=int)
    p.add_argument("--max_nodes", default=5000, type=int)
    p.add_argument("--threshold", default=0.05, type=float)
    p.add_argument("--gpu", action="store_true")
    args = p.parse_args()

    input_csv = Path(args.input_csv)
    target_mgf = Path(args.target_mgf)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        raise FileNotFoundError(f"input_csv not found: {input_csv}")
    if not target_mgf.exists():
        raise FileNotFoundError(f"target_mgf not found: {target_mgf}")

    mgf_entries = parse_mgf_titles(target_mgf)
    if not mgf_entries:
        raise RuntimeError(f"No spectra parsed from MGF: {target_mgf}")

    df = pd.read_csv(input_csv)
    # accept 'SMILES' as alias for 'smiles'
    if "smiles" not in df.columns and "SMILES" in df.columns:
        df = df.rename(columns={"SMILES": "smiles"})
    if "identifier" not in df.columns or "smiles" not in df.columns:
        raise ValueError("CSV must contain columns: identifier, smiles")

    # Build candidate labels.tsv for prediction
    labels_rows = []
    cand_index_rows = []

    for _, row in df.iterrows():
        identifier = _safe_strip(row["identifier"])
        true_smiles = _safe_strip(row["smiles"])
        if not identifier:
            continue
        if identifier not in mgf_entries:
            continue

        cands = _iter_candidate_smiles(row)
        if not cands:
            continue

        pepmass = float(mgf_entries[identifier].pepmass)
        for j, smi in enumerate(cands):
            spec_name = f"{identifier}__cand{j:02d}"
            labels_rows.append(
                {
                    "spec": spec_name,
                    "smiles": smi,
                    "inchikey": common.inchikey_from_smiles(smi),
                    "ionization": "[M+H]+",
                    "precursor": pepmass,
                    "instrument": "Orbitrap",
                    "collision_energies": "['30.0']",
                }
            )
            cand_index_rows.append(
                {
                    "identifier": identifier,
                    "spec": spec_name,
                    "cand_rank": j,
                    "cand_smiles": smi,
                    "true_smiles": true_smiles,
                    "is_true": int(smi == true_smiles),
                }
            )

    if not labels_rows:
        raise RuntimeError("No valid rows found: identifiers must exist in the target MGF.")

    labels_tsv = out_dir / "candidates_labels.tsv"
    pd.DataFrame(labels_rows).to_csv(labels_tsv, sep="\t", index=False)
    pd.DataFrame(cand_index_rows).to_csv(out_dir / "candidates_index.csv", index=False)

    # Run prediction using existing script (no source modification)
    pred_dir = out_dir / "preds"
    pred_dir.mkdir(parents=True, exist_ok=True)

    pred_cmd = [
        sys.executable,
        str(_SCRIPT_DIR / "predict_smis_trained_inten.py"),
        "--gen-checkpoint",
        str(args.gen_checkpoint),
        "--inten-checkpoint",
        str(args.inten_checkpoint),
        "--dataset-labels",
        str(labels_tsv),
        "--save-dir",
        str(pred_dir),
        "--sparse-out",
        "--sparse-k",
        str(args.sparse_k),
        "--num-workers",
        str(args.num_workers),
        "--batch-size",
        str(args.batch_size),
        "--max-nodes",
        str(args.max_nodes),
        "--threshold",
        str(args.threshold),
        "--adduct-shift",
    ]
    if args.gpu:
        pred_cmd.append("--gpu")

    (out_dir / "predict_cmd.txt").write_text(" ".join(pred_cmd) + "\n")
    subprocess.run(pred_cmd, check=True)

    preds_hdf5 = pred_dir / "preds.hdf5"
    if not preds_hdf5.exists():
        raise FileNotFoundError(f"Prediction output not found: {preds_hdf5}")

    # Prepare true spectra (binned) for cosine, fixed CE=30
    true_binned: Dict[str, np.ndarray] = {}
    for ident, e in mgf_entries.items():
        b = common.bin_spectra([e.peaks], num_bins=args.num_bins, upper_limit=args.upper_limit)[0]
        true_binned[ident] = b.astype(np.float32, copy=False)

    cand_df = pd.DataFrame(cand_index_rows)
    results = []
    for ident, group in cand_df.groupby("identifier", sort=False):
        tvec = true_binned.get(ident)
        if tvec is None:
            continue

        scored = []
        for r in group.itertuples(index=False):
            pred_dict = load_pred_binned_from_preds_hdf5(preds_hdf5, r.spec)
            pvec = pred_dict.get("30")
            if pvec is None:
                sim = float("nan")
            else:
                sim = cosine_similarity(pvec, tvec)
            scored.append((sim, int(r.cand_rank), str(r.cand_smiles), str(r.spec), int(r.is_true), str(r.true_smiles)))

        scored.sort(key=lambda x: (-1e9 if math.isnan(x[0]) else -x[0], x[1]))
        best_sim, best_rank, best_smi, best_spec, _, true_smi = scored[0]

        results.append(
            {
                "identifier": ident,
                "true_smiles": true_smi,
                "best_smiles": best_smi,
                "best_cand_rank": best_rank,
                "best_cosine": best_sim,
                "is_correct_top1": int(best_smi == true_smi),
                "true_in_candidates": int(any(x[4] == 1 for x in scored)),
                "num_candidates": int(len(scored)),
            }
        )

    res_df = pd.DataFrame(results)
    out_csv = out_dir / "end2end_pubchem_candidates_top1.csv"
    res_df.to_csv(out_csv, index=False)

    overall_acc = float(res_df["is_correct_top1"].mean()) if len(res_df) else 0.0
    subset = res_df[res_df["true_in_candidates"] == 1]
    acc_when_present = float(subset["is_correct_top1"].mean()) if len(subset) else 0.0

    summary = {
        "n_rows_scored": int(len(res_df)),
        "overall_top1_accuracy": overall_acc,
        "top1_accuracy_when_true_in_candidates": acc_when_present,
        "output_csv": str(out_csv),
        "preds_hdf5": str(preds_hdf5),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

