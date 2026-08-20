#!/usr/bin/env python3
"""This code was obtained from the ICEBERG implementation developed by
Goldman et al.

Original repository:
https://github.com/samgoldman97/ms-pred

No modifications were made to the original code.

The original copyright notice and license terms are retained.
"""

import argparse, csv, math, subprocess, sys
from pathlib import Path
from typing import Dict, List, Tuple
import h5py
import pandas as pd
from rdkit import Chem, RDLogger
import numpy as np

RDLogger.DisableLog("rdApp.*")

_MS_PRED_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_PREDICT_SCRIPT = _MS_PRED_ROOT / "dag_pred" / "predict_smis_trained_inten.py"

def canon(s: str) -> str:
    if not isinstance(s, str):
        return ""
    try:
        m = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(m, canonical=True) if m else ""
    except Exception:
        return ""

def parse_args():
    p = argparse.ArgumentParser(description="End-to-end: build labels from CSV+MGF, run ICEBERG prediction, evaluate Top1 by cosine.")
    p.add_argument("--csv", type=Path, required=True, help="pubchem_top100_by_ecfp4_bundle.csv")
    p.add_argument("--mgf", type=Path, required=True, help="MGF used for truth spectra and metadata")
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    p.add_argument("--gen-checkpoint", type=Path, required=True)
    p.add_argument("--inten-checkpoint", type=Path, required=True)
    p.add_argument(
        "--predict-script",
        type=Path,
        default=_DEFAULT_PREDICT_SCRIPT,
        help="predict_smis_trained_inten.py (default: next to dag_pred under this ms_pred tree)",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--sparse-k", type=int, default=100)
    p.add_argument("--max-nodes", type=int, default=5000)
    p.add_argument("--threshold", type=float, default=0.05)
    p.add_argument("--mz-tol-da", type=float, default=0.01, help="Cosine matching tolerance in Dalton")
    p.add_argument("--python", type=str, default=sys.executable, help="Python executable for predictor")
    p.add_argument("--gpu", action="store_true", help="Use GPU in predictor if available")
    return p.parse_args()

def read_mgf_metadata_and_truth(mgf: Path) -> Tuple[Dict[str, Tuple[str, str, str, str]], Dict[str, List[Tuple[float, float]]]]:
    meta: Dict[str, Tuple[str, str, str, str]] = {}
    truth: Dict[str, List[Tuple[float, float]]] = {}
    with mgf.open("r", encoding="utf-8", errors="ignore") as f:
        # Use None to mark "not inside a block"; a dict indicates an active block
        cur = None
        peaks: List[Tuple[float, float]] = []
        def add_current():
            if not cur:
                return
            ident = cur.get("TITLE") or cur.get("IDENTIFIER") or cur.get("FEATURE_ID")
            if ident:
                try:
                    # Prefer PEPMASS, but fall back to PRECURSOR_MZ if absent
                    pmz_src = cur.get("PEPMASS", "") or cur.get("PRECURSOR_MZ", "")
                    pmz = (pmz_src.split()[0] or "").strip()
                except Exception:
                    pmz = ""
                ion = cur.get("IONMODE", "") or ""
                inst = cur.get("INSTRUMENT", "Orbitrap") or "Orbitrap"
                ce = cur.get("COLLISION_ENERGIES", "['30.0']")
                meta[ident] = (pmz, ion, inst, ce)
                if peaks:
                    truth[ident] = peaks[:]
        for line in f:
            s = line.strip()
            if s == "BEGIN IONS":
                cur = {}
                peaks = []
            elif s == "END IONS":
                add_current()
                cur = None
                peaks = []
            else:
                if cur is None:
                    continue
                if "=" in s and not s[:1].isdigit():
                    k, v = s.split("=", 1)
                    cur[k.strip().upper()] = v.strip()
                else:
                    # peak line
                    try:
                        mz, inten = s.split()[:2]
                        peaks.append((float(mz), float(inten)))
                    except Exception:
                        pass
    return meta, truth

def write_candidates_labels(csv_in: Path, mgf_meta: Dict[str, Tuple[str, str, str, str]], out_tsv: Path):
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    idx: Dict[str, int] = {}
    with csv_in.open("r") as f, out_tsv.open("w", newline="") as g:
        r = csv.DictReader(f)
        w = csv.writer(g, delimiter="\t")
        w.writerow(["spec", "smiles", "inchikey", "ionization", "precursor", "instrument", "collision_energies"])
        for row in r:
            ident = row["identifier"]
            smi = row["candidate_smiles"]
            c = idx.get(ident, 0)
            idx[ident] = c + 1
            pmz, _ion_unused, inst, _ce_unused = mgf_meta.get(ident, ("", "", "Orbitrap", "['30.0']"))
            # Force ionization and CE as requested (use predictor-friendly token)
            ion_fixed = "[M+H]+"
            ce_fixed = "['30.0']"
            w.writerow([f"{ident}__cand{c:02d}", smi, "", ion_fixed, pmz, inst, ce_fixed])

def run_predictor(args, labels_tsv: Path, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.python, str(args.predict_script),
        "--gen-checkpoint", str(args.gen_checkpoint),
        "--inten-checkpoint", str(args.inten_checkpoint),
        "--dataset-labels", str(labels_tsv),
        "--save-dir", str(save_dir),
        "--sparse-out", "--sparse-k", str(args.sparse_k),
        "--num-workers", str(args.num_workers),
        "--batch-size", str(args.batch_size),
        "--max-nodes", str(args.max_nodes),
        "--threshold", str(args.threshold),
        "--adduct-shift",
    ]
    if args.gpu:
        cmd.insert(2, "--gpu")
    subprocess.run(cmd, check=True)
    return save_dir / "preds.hdf5"

def _bin_spectra_local(peaks_list: List[List[Tuple[float, float]]], num_bins: int, upper_limit: float) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    if num_bins <= 0 or upper_limit <= 0:
        return [np.zeros(1, dtype=np.float32) for _ in peaks_list]
    for peaks in peaks_list:
        v = np.zeros(num_bins, dtype=np.float32)
        for mz, inten in peaks:
            if mz <= 0 or mz > upper_limit:
                continue
            idx = int((mz / upper_limit) * num_bins)
            if 0 <= idx < num_bins:
                v[idx] += float(inten)
        out.append(v)
    return out

def _cosine_vec(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-22
    return float(np.dot(a, b) / denom)

def _load_pred_binned_from_h5(preds_h5: h5py.File, spec: str) -> Dict[str, np.ndarray]:
    """
    Returns: dict ce_key -> dense vector aligned with HDF5 attrs (num_bins, upper_limit).
    """
    out: Dict[str, np.ndarray] = {}
    pred_key = f"pred_{spec}"
    if pred_key not in preds_h5:
        return out
    root = preds_h5
    sparse_out = bool(root.attrs.get("sparse_out", True))
    num_bins = int(root.attrs.get("num_bins", 15000))
    upper_limit = float(root.attrs.get("upper_limit", 1500.0))
    grp = preds_h5[pred_key]
    # descend first level (ikey) then collision energies
    for _, ikey_grp in grp.items():
        if not isinstance(ikey_grp, h5py.Group):
            continue
        for ce_name, ce_grp in ikey_grp.items():
            if not isinstance(ce_grp, h5py.Group) or "spec" not in ce_grp:
                continue
            data = ce_grp["spec"][()]
            # Normalize to dense vector
            if data.ndim == 2 and data.shape[1] >= 2 and sparse_out:
                # data is (n_peaks, 2) with (mz, intensity) or (bin_idx, intensity)
                # We assume they are (mz, inten) and bin with same setting as predictor output attrs
                peaks = [(float(m), float(I)) for m, I in data]
                vec = _bin_spectra_local([peaks], num_bins=num_bins, upper_limit=upper_limit)[0]
            elif data.ndim == 1:
                vec = np.asarray(data, dtype=np.float32)
            else:
                # Fallback binning
                try:
                    peaks = [(float(m), float(I)) for m, I in data]
                except Exception:
                    peaks = []
                vec = _bin_spectra_local([peaks], num_bins=num_bins, upper_limit=upper_limit)[0]
            # normalize CE key to integer string like "30"
            try:
                ce_val = float(str(ce_name).replace("collision", "").strip())
            except Exception:
                ce_val = 30.0
            out[f"{ce_val:.0f}"] = vec
        break
    return out

def load_pred_for_spec(h5: h5py.File, spec: str):
    grp = f"pred_{spec}"
    if grp not in h5:
        return None
    def find_spec(o):
        if isinstance(o, h5py.Dataset) and o.name.endswith("/spec"):
            return o[()]
        if isinstance(o, h5py.Group):
            for _, v in o.items():
                r = find_spec(v)
                if r is not None:
                    return r
        return None
    return find_spec(h5[grp])

def evaluate_top1(csv_in: Path, preds_h5: Path, truth: Dict[str, List[Tuple[float, float]]], mz_tol_da: float) -> Tuple[int, int]:
    from collections import defaultdict
    true_by_ident: Dict[str, str] = {}
    cands_by_ident: Dict[str, List[str]] = defaultdict(list)
    with csv_in.open("r") as f:
        r = csv.DictReader(f)
        for row in r:
            ident = row["identifier"]
            if ident not in true_by_ident:
                true_by_ident[ident] = row["true_smiles"]
            cands_by_ident[ident].append(row["candidate_smiles"])
    ok = tot = 0
    with h5py.File(preds_h5, "r") as h5:
        # Use predictor binning spec for truth vectors
        num_bins = int(h5.attrs.get("num_bins", 15000))
        upper_limit = float(h5.attrs.get("upper_limit", 1500.0))
        true_vec_by_ident: Dict[str, np.ndarray] = {}
        if truth:
            # Pre-bin all available truth spectra using predictor settings
            for ident, peaks in truth.items():
                vec = _bin_spectra_local([peaks], num_bins=num_bins, upper_limit=upper_limit)[0]
                true_vec_by_ident[ident] = vec
        for ident, cand_list in cands_by_ident.items():
            tvec = true_vec_by_ident.get(ident)
            if tvec is None:
                continue
            sims: List[Tuple[float, int]] = []
            for i, _ in enumerate(cand_list):
                spec = f"{ident}__cand{i:02d}"
                pred_dict = _load_pred_binned_from_h5(h5, spec)
                # Prefer 30 eV if available, else pick any
                pvec = pred_dict.get("30")
                if pvec is None and pred_dict:
                    # take the first CE available
                    pvec = next(iter(pred_dict.values()))
                if pvec is None:
                    sims.append((-1e-9, i))
                else:
                    sims.append((_cosine_vec(tvec, pvec), i))
            if not sims:
                continue
            sims.sort(reverse=True)
            best_idx = sims[0][1]
            best = cand_list[best_idx]
            if canon(best) and canon(best) == canon(true_by_ident.get(ident, "")):
                ok += 1
            tot += 1
    return ok, tot

def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    labels_tsv = args.out_dir / "candidates_labels.tsv"
    preds_dir = args.out_dir / "preds"

    print("Reading MGF for metadata and truth spectra ...")
    mgf_meta, truth = read_mgf_metadata_and_truth(args.mgf)
    print(f"MGF entries parsed: meta={len(mgf_meta)}, truth={len(truth)}")

    print("Building candidates_labels.tsv ...")
    write_candidates_labels(args.csv, mgf_meta, labels_tsv)
    print(f"Wrote: {labels_tsv}")

    print("Running ICEBERG predictor ...")
    preds_h5 = run_predictor(args, labels_tsv, preds_dir)
    print(f"Predictions HDF5: {preds_h5}")

    print("Evaluating Top-1 (canonical SMILES) by cosine ...")
    ok, tot = evaluate_top1(args.csv, preds_h5, truth, args.mz_tol_da)
    acc = (ok / tot * 100.0) if tot else 0.0
    summary_path = args.out_dir / "top1_summary.csv"
    pd.DataFrame([{"top1_correct": ok, "total": tot, "top1_acc_pct": acc}]).to_csv(summary_path, index=False)
    print(f"Top1(true@1, canonical): {ok}/{tot} ({acc:.2f}%)")
    print(f"Wrote summary: {summary_path}")

if __name__ == "__main__":
    main()

