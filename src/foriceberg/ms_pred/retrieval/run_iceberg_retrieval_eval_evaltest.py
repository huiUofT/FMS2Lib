#!/usr/bin/env python3
"""This code was obtained from the ICEBERG implementation developed by
Goldman et al.

Original repository:
https://github.com/samgoldman97/ms-pred

No modifications were made to the original code.

The original copyright notice and license terms are retained.

Same pipeline as run_iceberg_retrieval_eval.py, but scoring uses evaluate_test_set-style
dist_bin cosine; writes per-spectrum rows to per_spectrum_retrieval.tsv in addition to
top1_summary.csv.
"""
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd

from . import run_iceberg_retrieval_eval as base


def dist_bin(
    cand_preds_dict,
    true_spec_dict,
    sparse=True,
    ignore_peak=None,
    func="cos",
    selected_evs=None,
    agg=True,
):
    """evaluate_test_set.py-compatible cosine distance."""
    dist = []
    true_npeaks = []

    if selected_evs:
        true_spec_dict = {k: v for k, v in true_spec_dict.items() if str(k) in selected_evs}

    true_spec_dict = {f"{float(k):.0f}": v for k, v in true_spec_dict.items()}
    cand_preds_dict = [{f"{float(k):.0f}": v for k, v in cand_dict.items()} for cand_dict in cand_preds_dict]

    for colli_eng in true_spec_dict.keys():
        true_spec = true_spec_dict[colli_eng]
        cand_preds = np.stack([i.get(colli_eng, np.zeros_like(true_spec)) for i in cand_preds_dict], axis=0)
        pred_specs = cand_preds if not sparse else np.zeros((len(cand_preds), true_spec.shape[0]))

        if sparse:
            inds = cand_preds[:, :, 0].astype(int)
            pos_1 = np.repeat(np.arange(inds.shape[0])[:, np.newaxis], inds.shape[1], axis=1)
            valid_mask = inds < pred_specs.shape[1]
            pred_specs[pos_1[valid_mask], inds[valid_mask]] = cand_preds[:, :, 1][valid_mask]

        if ignore_peak:
            pred_specs[:, int(ignore_peak):] = 0
            true_spec = true_spec.copy()
            true_spec[int(ignore_peak):] = 0

        true_npeaks.append(np.sum(true_spec > 0))

        if func == "cos":
            norm_pred = np.linalg.norm(pred_specs, axis=-1) + 1e-22
            norm_true = np.linalg.norm(true_spec, axis=-1) + 1e-22
            dist.append(1 - np.dot(pred_specs, true_spec) / (norm_pred * norm_true))
        else:
            raise ValueError(f"Unsupported func: {func}")

    dist = np.array(dist)
    weights = (np.array(true_npeaks) >= 5) * 3 + (np.array(true_npeaks) >= 1) * 1
    weights = weights / (weights.sum() + 1e-9)
    if agg:
        return np.sum(dist * weights[:, None], axis=0)
    dist = dist[weights > 0]
    return dist, np.sum(dist * weights[:, None], axis=0)


def evaluate_top1_evaltest_detailed(
    csv_in: Path,
    preds_h5: Path,
    truth: Dict[str, List[Tuple[float, float]]],
) -> Tuple[int, int, List[Dict[str, Any]]]:
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
    per_rows: List[Dict[str, Any]] = []

    with h5py.File(preds_h5, "r") as h5:
        num_bins = int(h5.attrs.get("num_bins", 15000))
        upper_limit = float(h5.attrs.get("upper_limit", 1500.0))
        true_vec_by_ident: Dict[str, np.ndarray] = {}
        for ident, peaks in truth.items():
            vec = base._bin_spectra_local([peaks], num_bins=num_bins, upper_limit=upper_limit)[0]
            true_vec_by_ident[ident] = vec

        for ident, cand_list in cands_by_ident.items():
            ts = true_by_ident.get(ident, "")
            tvec = true_vec_by_ident.get(ident)

            if tvec is None:
                per_rows.append(
                    {
                        "identifier": ident,
                        "true_smiles": ts,
                        "n_candidates": len(cand_list),
                        "top1_hit": "",
                        "true_rank": "",
                        "similarity_top1": "",
                        "similarity_true_candidate": "",
                        "note": "no_experimental_spectrum_in_mgf",
                    }
                )
                continue

            sims: List[Tuple[float, int]] = []
            for i, _ in enumerate(cand_list):
                spec = f"{ident}__cand{i:02d}"
                pred_dict = base._load_pred_binned_from_h5(h5, spec)
                if not pred_dict:
                    sims.append((-1e-9, i))
                    continue
                d = dist_bin(
                    cand_preds_dict=[pred_dict],
                    true_spec_dict={"30": tvec},
                    sparse=False,
                    func="cos",
                    selected_evs=["30"],
                    agg=True,
                )
                sims.append((float(1.0 - d[0]), i))

            if not sims:
                per_rows.append(
                    {
                        "identifier": ident,
                        "true_smiles": ts,
                        "n_candidates": len(cand_list),
                        "top1_hit": "",
                        "true_rank": "",
                        "similarity_top1": "",
                        "similarity_true_candidate": "",
                        "note": "no_scores",
                    }
                )
                continue

            sims.sort(reverse=True, key=lambda x: (x[0], -x[1]))
            best_idx = sims[0][1]
            best = cand_list[best_idx]
            hit = bool(base.canon(best) and base.canon(best) == base.canon(ts))
            sim_top = sims[0][0]
            ok += int(hit)
            tot += 1

            true_idx: int | None = None
            t_canon = base.canon(ts)
            for i, c in enumerate(cand_list):
                if base.canon(c) == t_canon:
                    true_idx = i
                    break

            true_rank: str | int = ""
            sim_true: str | float = ""
            if true_idx is not None:
                for r, (s, idx) in enumerate(sims):
                    if idx == true_idx:
                        true_rank = r + 1
                        sim_true = s
                        break
                note = ""
            else:
                note = "true_smiles_not_among_candidates"

            per_rows.append(
                {
                    "identifier": ident,
                    "true_smiles": ts,
                    "n_candidates": len(cand_list),
                    "top1_hit": int(hit),
                    "true_rank": true_rank,
                    "similarity_top1": sim_top,
                    "similarity_true_candidate": sim_true,
                    "note": note,
                }
            )

    return ok, tot, per_rows


def main():
    args = base.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    labels_tsv = args.out_dir / "candidates_labels.tsv"
    preds_dir = args.out_dir / "preds"

    print("Reading MGF for metadata and truth spectra ...")
    mgf_meta, truth = base.read_mgf_metadata_and_truth(args.mgf)
    print(f"MGF entries parsed: meta={len(mgf_meta)}, truth={len(truth)}")

    print("Building candidates_labels.tsv ...")
    base.write_candidates_labels(args.csv, mgf_meta, labels_tsv)
    print(f"Wrote: {labels_tsv}")

    print("Running ICEBERG predictor ...")
    preds_h5 = base.run_predictor(args, labels_tsv, preds_dir)
    print(f"Predictions HDF5: {preds_h5}")

    print("Evaluating Top-1 (dist_bin) + per-spectrum table ...")
    ok, tot, per_rows = evaluate_top1_evaltest_detailed(args.csv, preds_h5, truth)
    acc = (ok / tot * 100.0) if tot else 0.0
    summary_path = args.out_dir / "top1_summary.csv"
    pd.DataFrame([{"top1_correct": ok, "total": tot, "top1_acc_pct": acc}]).to_csv(summary_path, index=False)
    per_path = args.out_dir / "per_spectrum_retrieval.tsv"
    pd.DataFrame(per_rows).to_csv(per_path, sep="\t", index=False)
    print(f"Top1(true@1, canonical): {ok}/{tot} ({acc:.2f}%)")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote per-spectrum: {per_path}")


if __name__ == "__main__":
    main()
