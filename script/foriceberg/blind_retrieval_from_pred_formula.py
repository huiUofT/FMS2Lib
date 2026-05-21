#!/usr/bin/env python3
"""
Blind structure retrieval: predicted formula (from CSV) + experimental MS/MS (MGF peaks only
for scoring). True SMILES in the MGF is never used to build or rank candidates.

Pipeline:
  1) Read per-spectrum predicted formula from CSV (e.g. MS1/MS2 ranker output).
  2) Collect SMILES candidates by that formula from SusDat CSV and/or PubChem-by-formula HDF5.
  3) Run ICEBERG (gen + inten checkpoints under mode/foriceberg by default) on all candidates.
  4) Score each candidate vs the experimental spectrum using the same binned cosine as
     ms_pred.retrieval.run_iceberg_retrieval_eval_evaltest.dist_bin at 30 eV.
  5) Write result CSV with formula, SMILES, candidate source, similarity, etc.

Uses ms_pred.retrieval (src/foriceberg) for MGF IO, label building, predictor wiring, and dist_bin.
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from foriceberg_paths import github_root, mode_dir, pythonpath_src, repo_root, script_dir, src_dir

_SCRIPT_DIR = script_dir()
_GITHUB_ROOT = repo_root()
_GITHUB_SRC = src_dir()

if str(_GITHUB_SRC) not in sys.path:
    sys.path.insert(0, str(_GITHUB_SRC))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from build_pubchem_top100_by_ecfp4_from_mgf import (  # noqa: E402
    normalize_formula,
    read_pubchem_smiles_for_formula,
)

from ms_pred.retrieval import run_iceberg_retrieval_eval as base  # noqa: E402
from ms_pred.retrieval.run_iceberg_retrieval_eval_evaltest import dist_bin  # noqa: E402

_DEFAULT_GEN = mode_dir() / "generation" / "best.ckpt"
_DEFAULT_INTEN = mode_dir() / "intensity" / "best.ckpt"
_DEFAULT_PRED = src_dir() / "ms_pred" / "dag_pred" / "predict_smis_trained_inten.py"


def canon(s: str) -> str:
    try:
        m = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(m, canonical=True) if m else ""
    except Exception:
        return ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Blind retrieval: pred formula CSV + MGF (peaks only) + SusDat / PubChem HDF5 + ICEBERG."
    )
    p.add_argument("--mgf", type=Path, required=True, help="Experimental MGF (SMILES in header is ignored for ranking)")
    p.add_argument(
        "--pred-csv",
        type=Path,
        required=True,
        help="CSV with identifier and predicted formula (columns: identifier, pred_formula or predicted_formula)",
    )
    p.add_argument(
        "--susdat",
        type=Path,
        default=None,
        help="SusDat-style CSV (formula, Name, SMILES). Optional if --pubchem-hdf5 is set.",
    )
    p.add_argument(
        "--pubchem-hdf5",
        type=Path,
        default=None,
        help="PubChem-by-formula HDF5 (same layout as build_pubchem_top100_by_ecfp4_from_mgf). Optional if --susdat is set.",
    )
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory (labels, preds, tables)")
    p.add_argument("--gen-checkpoint", type=Path, default=_DEFAULT_GEN)
    p.add_argument("--inten-checkpoint", type=Path, default=_DEFAULT_INTEN)
    p.add_argument("--predict-script", type=Path, default=_DEFAULT_PRED)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--sparse-k", type=int, default=100)
    p.add_argument("--max-nodes", type=int, default=5000)
    p.add_argument("--threshold", type=float, default=0.05)
    p.add_argument("--gpu", action="store_true")
    p.add_argument(
        "--max-candidates",
        type=int,
        default=400,
        help="Cap merged candidates per spectrum after dedup (0 = no cap). Order: SusDat first, then PubChem.",
    )
    p.add_argument(
        "--write-ranking-tsv",
        action="store_true",
        help="Also write per_spectrum_all_candidates.tsv (all scores).",
    )
    p.add_argument(
        "--audit-mgf-smiles",
        action="store_true",
        help="Add optional column mgf_smiles_audit from MGF header (for offline checking only).",
    )
    p.add_argument("--skip-predict", action="store_true", help="Stop after writing labels (debug).")
    p.add_argument("--python", type=str, default=sys.executable)
    return p.parse_args()


def load_pred_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    cols = {c.lower(): c for c in df.columns}
    if "identifier" not in cols:
        raise ValueError("pred CSV must contain column 'identifier'")
    id_col = cols["identifier"]
    form_col = None
    for key in ("pred_formula", "predicted_formula", "formula"):
        if key in cols:
            form_col = cols[key]
            break
    if form_col is None:
        raise ValueError("pred CSV needs pred_formula / predicted_formula / formula column")
    pmz_col = cols.get("precursor_mz") or cols.get("precursor mz")
    out = pd.DataFrame(
        {
            "identifier": df[id_col].astype(str),
            "pred_formula_raw": df[form_col].astype(str),
        }
    )
    if pmz_col:
        out["precursor_mz_csv"] = df[pmz_col].astype(str)
    else:
        out["precursor_mz_csv"] = ""
    return out


def load_susdat_by_formula(path: Path) -> Dict[str, List[Tuple[str, str]]]:
    """norm_formula -> list of (raw_smiles, name)."""
    df = pd.read_csv(path, low_memory=False)
    if "formula" not in df.columns and "Molecular_Formula" in df.columns:
        df = df.rename(columns={"Molecular_Formula": "formula"})
    need = {"formula", "SMILES"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"susdat missing columns: {miss}")
    name_col = "Name" if "Name" in df.columns else None
    out: Dict[str, List[Tuple[str, str]]] = {}
    for rec in df.itertuples(index=False):
        nf = normalize_formula(str(getattr(rec, "formula")))
        smi = str(getattr(rec, "SMILES")).strip()
        if not nf or not smi:
            continue
        nm = str(getattr(rec, name_col, "")).strip() if name_col else ""
        out.setdefault(nf, []).append((smi, nm))
    return out


def merge_sources_label(src: frozenset) -> str:
    parts = sorted(src)
    return "+".join(parts)


def collect_candidates_for_formula(
    norm_f: str,
    raw_f: str,
    susdat_by_norm: Optional[Dict[str, List[Tuple[str, str]]]],
    pubchem_h5: Optional[h5py.File],
    max_cands: int,
) -> List[Tuple[str, str, str, str]]:
    """
    Returns list of (raw_smiles, source_label, susdat_name_or_empty, canonical_smiles_key)
    deduplicated by canonical SMILES, SusDat rows first then PubChem.
    source_label is susdat | pubchem | susdat+pubchem
    """
    # canonical -> (raw_smiles, sources set, susdat_name)
    merged: Dict[str, Tuple[str, frozenset, str]] = {}

    def add(smi: str, src: str, name: str) -> None:
        c = canon(smi)
        if not c:
            return
        if c not in merged:
            merged[c] = (smi, frozenset([src]), name if src == "susdat" else "")
        else:
            raw0, fs, nm0 = merged[c]
            ns = set(fs)
            ns.add(src)
            nm = nm0 or (name if src == "susdat" else "")
            merged[c] = (raw0, frozenset(ns), nm)

    if susdat_by_norm is not None:
        for smi, nm in susdat_by_norm.get(norm_f, []):
            add(smi, "susdat", nm)

    if pubchem_h5 is not None:
        for smi in read_pubchem_smiles_for_formula(pubchem_h5, norm_f, raw_f):
            add(smi, "pubchem", "")

    # stable order: susdat-first by first appearance in merged values — rebuild order list
    ordered: List[str] = []
    seen: set = set()

    def push_from_pairs(pairs: Sequence[Tuple[str, str]]) -> None:
        for smi, nm in pairs:
            c = canon(smi)
            if c and c not in seen and c in merged:
                seen.add(c)
                ordered.append(c)

    if susdat_by_norm is not None:
        push_from_pairs(susdat_by_norm.get(norm_f, []))
    if pubchem_h5 is not None:
        for smi in read_pubchem_smiles_for_formula(pubchem_h5, norm_f, raw_f):
            c = canon(smi)
            if c and c not in seen and c in merged:
                seen.add(c)
                ordered.append(c)

    # any keys left (should not happen)
    for c in merged:
        if c not in seen:
            ordered.append(c)

    rows: List[Tuple[str, str, str, str]] = []
    for c in ordered:
        raw_s, fs, nm = merged[c]
        if max_cands and len(rows) >= max_cands:
            break
        rows.append((raw_s, merge_sources_label(fs), nm, c))
    return rows


def extract_mgf_smiles_audit(mgf_path: Path, ident: str) -> str:
    """Read SMILES= from MGF for audit only (not used for candidates)."""
    want = ident.strip()
    cur: Optional[dict] = None
    with mgf_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s == "BEGIN IONS":
                cur = {}
            elif s == "END IONS":
                if cur:
                    tid = cur.get("TITLE") or cur.get("IDENTIFIER") or cur.get("FEATURE_ID")
                    if tid == want:
                        return str(cur.get("SMILES", "") or "")
                cur = None
            elif cur is not None and "=" in s and not s[0].isdigit():
                k, v = s.split("=", 1)
                cur[k.strip().upper()] = v.strip()
    return ""


def run_predictor_local(args: argparse.Namespace, labels_tsv: Path, save_dir: Path) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.python,
        str(args.predict_script),
        "--gen-checkpoint",
        str(args.gen_checkpoint),
        "--inten-checkpoint",
        str(args.inten_checkpoint),
        "--dataset-labels",
        str(labels_tsv),
        "--save-dir",
        str(save_dir),
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
        cmd.insert(2, "--gpu")
    env = os.environ.copy()
    env["PYTHONPATH"] = pythonpath_src() + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, cwd=str(_GITHUB_ROOT), env=env, check=True)
    out_h5 = save_dir / "preds.hdf5"
    if not out_h5.is_file():
        raise FileNotFoundError(f"Predictor did not write {out_h5}")
    return out_h5


def score_candidate(
    h5: h5py.File,
    spec: str,
    tvec: np.ndarray,
) -> float:
    pred_dict = base._load_pred_binned_from_h5(h5, spec)
    if not pred_dict:
        return float("-inf")
    d = dist_bin(
        cand_preds_dict=[pred_dict],
        true_spec_dict={"30": tvec},
        sparse=False,
        func="cos",
        selected_evs=["30"],
        agg=True,
    )
    dist_v = float(np.asarray(d).ravel()[0])
    return 1.0 - dist_v


def main() -> None:
    args = parse_args()
    if not args.susdat and not args.pubchem_hdf5:
        sys.exit("Provide at least one of --susdat or --pubchem-hdf5")
    for lab, p in (("GEN", args.gen_checkpoint), ("INTEN", args.inten_checkpoint)):
        if not p.is_file():
            sys.exit(f"Missing {lab} checkpoint: {p}")
    if not args.predict_script.is_file():
        sys.exit(f"Missing predict script: {args.predict_script}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    pred_df = load_pred_table(args.pred_csv)
    mgf_meta, truth = base.read_mgf_metadata_and_truth(args.mgf)
    susdat_map = load_susdat_by_formula(args.susdat) if args.susdat else None

    pubchem_path = args.pubchem_hdf5
    h5_ctx = h5py.File(pubchem_path, "r") if pubchem_path else None

    cand_rows_for_labels: List[Dict[str, str]] = []
    per_ident_cands: Dict[str, List[Tuple[str, str, str]]] = {}
    skipped: List[Dict[str, str]] = []

    try:
        for rec in pred_df.itertuples(index=False):
            ident = str(rec.identifier).strip()
            raw_f = str(rec.pred_formula_raw).strip()
            norm_f = normalize_formula(raw_f)
            if not ident or not norm_f:
                skipped.append({"identifier": ident, "reason": "empty_id_or_formula"})
                continue
            if ident not in mgf_meta:
                skipped.append({"identifier": ident, "reason": "identifier_not_in_mgf"})
                continue

            cands = collect_candidates_for_formula(
                norm_f,
                raw_f,
                susdat_map,
                h5_ctx,
                args.max_candidates,
            )
            if not cands:
                skipped.append({"identifier": ident, "reason": "no_candidates_for_formula"})
                continue

            per_ident_cands[ident] = [(t[0], t[1], t[2]) for t in cands]  # smiles, source, susdat_name
            for smi, src, nm in per_ident_cands[ident]:
                cand_rows_for_labels.append(
                    {
                        "identifier": ident,
                        "candidate_smiles": smi,
                        "candidate_source": src,
                        "pred_formula_norm": norm_f,
                        "pred_formula_raw": raw_f,
                        "susdat_name": nm,
                    }
                )
    finally:
        if h5_ctx is not None:
            h5_ctx.close()

    pd.DataFrame(skipped).to_csv(args.out_dir / "skipped_spectra.csv", index=False)
    if not cand_rows_for_labels:
        print("No candidates to predict. See skipped_spectra.csv")
        return

    labels_csv = args.out_dir / "candidates_long.csv"
    pd.DataFrame(cand_rows_for_labels).to_csv(labels_csv, index=False)

    # Predictor expects two columns for write_candidates_labels
    compat = args.out_dir / "candidates_for_predictor.csv"
    with compat.open("w", newline="", encoding="utf-8") as g:
        w = csv.DictWriter(g, fieldnames=["identifier", "candidate_smiles"])
        w.writeheader()
        for r in cand_rows_for_labels:
            w.writerow({"identifier": r["identifier"], "candidate_smiles": r["candidate_smiles"]})

    labels_tsv = args.out_dir / "candidates_labels.tsv"
    base.write_candidates_labels(compat, mgf_meta, labels_tsv)
    print(f"Wrote {labels_tsv} ({sum(len(v) for v in per_ident_cands.values())} candidate rows)")

    if args.skip_predict:
        print("--skip-predict: done.")
        return

    preds_dir = args.out_dir / "preds"
    preds_h5 = run_predictor_local(args, labels_tsv, preds_dir)
    print(f"Predictions: {preds_h5}")

    results: List[Dict[str, object]] = []
    ranking_rows: List[Dict[str, object]] = []

    with h5py.File(preds_h5, "r") as h5:
        num_bins = int(h5.attrs.get("num_bins", 15000))
        upper_limit = float(h5.attrs.get("upper_limit", 1500.0))

        for rec in pred_df.itertuples(index=False):
            ident = str(rec.identifier).strip()
            if ident not in per_ident_cands:
                continue
            pmz_csv = str(getattr(rec, "precursor_mz_csv", "")).strip()
            raw_f = str(rec.pred_formula_raw).strip()
            norm_f = normalize_formula(raw_f)

            peaks = truth.get(ident)
            if not peaks:
                results.append(
                    {
                        "identifier": ident,
                        "precursor_mz": pmz_csv or (mgf_meta.get(ident, ("",))[0]),
                        "pred_formula": norm_f,
                        "pred_formula_raw": raw_f,
                        "best_smiles": "",
                        "candidate_source": "",
                        "susdat_name": "",
                        "best_similarity": "",
                        "n_candidates": len(per_ident_cands[ident]),
                        "note": "no_experimental_peaks_in_mgf",
                    }
                )
                continue

            tvec = base._bin_spectra_local([peaks], num_bins=num_bins, upper_limit=upper_limit)[0]
            best_i = -1
            best_sim = float("-inf")
            sims: List[Tuple[float, int]] = []

            for i in range(len(per_ident_cands[ident])):
                spec = f"{ident}__cand{i:02d}"
                sim = score_candidate(h5, spec, tvec)
                sims.append((sim, i))
                if sim > best_sim:
                    best_sim = sim
                    best_i = i

            smi, src, nm = per_ident_cands[ident][best_i]
            pmz_meta = mgf_meta.get(ident, ("",))[0]

            row = {
                "identifier": ident,
                "precursor_mz": pmz_meta or pmz_csv,
                "pred_formula": norm_f,
                "pred_formula_raw": raw_f,
                "best_smiles": smi,
                "candidate_source": src,
                "susdat_name": nm,
                "best_similarity": best_sim,
                "n_candidates": len(per_ident_cands[ident]),
                "note": "",
            }
            if args.audit_mgf_smiles:
                row["mgf_smiles_audit"] = extract_mgf_smiles_audit(args.mgf, ident)
            results.append(row)

            if args.write_ranking_tsv:
                sims_sorted = sorted(sims, key=lambda x: (-x[0], x[1]))
                for rank, (sim, i) in enumerate(sims_sorted, start=1):
                    s2, src2, nm2 = per_ident_cands[ident][i]
                    ranking_rows.append(
                        {
                            "identifier": ident,
                            "formula": norm_f,
                            "rank": rank,
                            "smiles": s2,
                            "candidate_source": src2,
                            "susdat_name": nm2,
                            "similarity": sim,
                        }
                    )

    out_csv = args.out_dir / "blind_retrieval_top1.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    if args.write_ranking_tsv and ranking_rows:
        rr = pd.DataFrame(ranking_rows)
        p = args.out_dir / "per_spectrum_all_candidates.tsv"
        rr.to_csv(p, sep="\t", index=False)
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
