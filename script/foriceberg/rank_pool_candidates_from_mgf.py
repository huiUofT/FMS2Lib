#!/usr/bin/env python3
"""
Rank a fixed candidate pool against experimental MGF spectra with ICEBERG.

This is a blind retrieval helper for cases where candidates are already listed
in a table. It never inserts or uses a true SMILES from the MGF; every spectrum
is scored against the same candidate pool.
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import xml.etree.ElementTree as ET

import h5py
import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from foriceberg_paths import mode_dir, repo_root, script_dir, src_dir

_SCRIPT_DIR = script_dir()
_GITHUB_ROOT = repo_root()
_GITHUB_SRC = src_dir()

if str(_GITHUB_SRC) not in sys.path:
    sys.path.insert(0, str(_GITHUB_SRC))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from ms_pred.retrieval import run_iceberg_retrieval_eval as base  # noqa: E402
from ms_pred.retrieval.run_iceberg_retrieval_eval_evaltest import dist_bin  # noqa: E402

_DEFAULT_GEN = mode_dir() / "generation" / "best.ckpt"
_DEFAULT_INTEN = mode_dir() / "intensity" / "best.ckpt"
_DEFAULT_PRED = src_dir() / "ms_pred" / "dag_pred" / "predict_smis_trained_inten.py"

_XLSX_NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rank fixed SMILES candidates from a table against MGF spectra; no true SMILES insertion."
    )
    p.add_argument("--mgf", type=Path, required=True, help="Experimental MGF; peaks are used for scoring")
    p.add_argument(
        "--candidates",
        "--candidates-xlsx",
        dest="candidates",
        type=Path,
        required=True,
        help="Candidate table (.xlsx, .csv, or .tsv) containing a SMILES/smile column",
    )
    p.add_argument(
        "--smiles-column",
        default=None,
        help="Candidate SMILES column. Default: auto-detect SMILES/smile/candidate_smiles.",
    )
    p.add_argument(
        "--identifier",
        action="append",
        default=None,
        help="Optional MGF identifier to score. Repeat to keep multiple IDs. Default: all spectra in MGF.",
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--gen-checkpoint", type=Path, default=_DEFAULT_GEN)
    p.add_argument("--inten-checkpoint", type=Path, default=_DEFAULT_INTEN)
    p.add_argument("--predict-script", type=Path, default=_DEFAULT_PRED)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--sparse-k", type=int, default=100)
    p.add_argument("--max-nodes", type=int, default=5000)
    p.add_argument("--threshold", type=float, default=0.05)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--python", type=str, default=sys.executable)
    p.add_argument("--skip-predict", action="store_true", help="Validate inputs and stop before prediction")
    p.add_argument(
        "--preds-h5",
        type=Path,
        default=None,
        help="Reuse an existing preds.hdf5 instead of running the predictor.",
    )
    return p.parse_args()


def canon_smiles(smiles: str) -> str:
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        return Chem.MolToSmiles(mol, canonical=True) if mol else ""
    except Exception:
        return ""


def _col_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha()).upper()
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


def _read_xlsx_rows(path: Path) -> List[Dict[str, str]]:
    """Read the first worksheet using only the Python standard library."""
    with zipfile.ZipFile(path) as zf:
        shared: List[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("a:si", _XLSX_NS):
                shared.append("".join(t.text or "" for t in si.findall(".//a:t", _XLSX_NS)))

        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rid_to_target = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
        sheet = workbook.find(".//a:sheet", _XLSX_NS)
        if sheet is None:
            return []
        rid = sheet.attrib[f"{{{_XLSX_NS['r']}}}id"]
        target = rid_to_target[rid]
        sheet_path = "xl/" + target.lstrip("/") if not target.startswith("xl/") else target
        sheet_root = ET.fromstring(zf.read(sheet_path))

        table_rows: List[List[str]] = []
        for row in sheet_root.findall(".//a:sheetData/a:row", _XLSX_NS):
            values: List[str] = []
            for cell in row.findall("a:c", _XLSX_NS):
                idx = _col_index(cell.attrib.get("r", "A"))
                while len(values) <= idx:
                    values.append("")
                cell_type = cell.attrib.get("t")
                value_node = cell.find("a:v", _XLSX_NS)
                if cell_type == "inlineStr":
                    text = "".join(t.text or "" for t in cell.findall(".//a:t", _XLSX_NS))
                elif value_node is None:
                    text = ""
                elif cell_type == "s":
                    text = shared[int(value_node.text or 0)]
                else:
                    text = value_node.text or ""
                values[idx] = text
            table_rows.append(values)

    if not table_rows:
        return []
    headers = [h.strip() for h in table_rows[0]]
    out: List[Dict[str, str]] = []
    for raw in table_rows[1:]:
        row = {headers[i]: (raw[i].strip() if i < len(raw) else "") for i in range(len(headers)) if headers[i]}
        if any(row.values()):
            out.append(row)
    return out


def _read_delimited_rows(path: Path) -> List[Dict[str, str]]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter=delimiter)]


def read_table_rows(path: Path) -> List[Dict[str, str]]:
    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        return _read_xlsx_rows(path)
    if suffix in {".csv", ".tsv"}:
        return _read_delimited_rows(path)
    raise ValueError(f"Unsupported candidate table type: {path.suffix}")


def find_smiles_column(rows: List[Dict[str, str]], requested: Optional[str]) -> str:
    if not rows:
        raise ValueError("Candidate table is empty")
    columns = list(rows[0].keys())
    if requested:
        for col in columns:
            if col == requested or col.lower() == requested.lower():
                return col
        raise ValueError(f"SMILES column not found: {requested}; available columns: {columns}")

    normalized = {col.strip().lower().replace(" ", "_"): col for col in columns}
    for key in ("smiles", "smile", "candidate_smiles"):
        if key in normalized:
            return normalized[key]
    raise ValueError(f"Could not auto-detect SMILES column; available columns: {columns}")


def load_candidate_pool(path: Path, smiles_column: Optional[str]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    raw_rows = read_table_rows(path)
    smi_col = find_smiles_column(raw_rows, smiles_column)
    candidates: List[Dict[str, str]] = []
    skipped: List[Dict[str, str]] = []
    seen: set[str] = set()

    for row_num, row in enumerate(raw_rows, start=2):
        raw_smiles = str(row.get(smi_col, "")).strip()
        can = canon_smiles(raw_smiles)
        if not raw_smiles:
            skipped.append({"row_number": str(row_num), "reason": "empty_smiles"})
            continue
        if not can:
            skipped.append({"row_number": str(row_num), "reason": "invalid_smiles", "smiles": raw_smiles})
            continue
        if can in seen:
            skipped.append({"row_number": str(row_num), "reason": "duplicate_canonical_smiles", "smiles": raw_smiles})
            continue
        seen.add(can)
        meta = {k: v for k, v in row.items() if k != smi_col}
        candidates.append(
            {
                "candidate_pool_index": str(len(candidates)),
                "smiles": raw_smiles,
                "canonical_smiles": can,
                **meta,
            }
        )
    return candidates, skipped


def run_predictor(args: argparse.Namespace, labels_tsv: Path, save_dir: Path) -> Path:
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
    env["PYTHONPATH"] = str(_GITHUB_SRC) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, cwd=str(_GITHUB_ROOT), env=env, check=True)
    preds_h5 = save_dir / "preds.hdf5"
    if not preds_h5.is_file():
        raise FileNotFoundError(f"Predictor did not write {preds_h5}")
    return preds_h5


def score_candidate(h5: h5py.File, spec: str, truth_vec: np.ndarray) -> float:
    pred_dict = base._load_pred_binned_from_h5(h5, spec)
    if not pred_dict:
        return float("-inf")
    dist = dist_bin(
        cand_preds_dict=[pred_dict],
        true_spec_dict={"30": truth_vec},
        sparse=False,
        func="cos",
        selected_evs=["30"],
        agg=True,
    )
    return 1.0 - float(np.asarray(dist).ravel()[0])


def write_dicts(path: Path, rows: Iterable[Dict[str, object]], delimiter: str = ",") -> None:
    materialized = list(rows)
    fieldnames: List[str] = []
    for row in materialized:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        for row in materialized:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    for label, path in (("MGF", args.mgf), ("candidate table", args.candidates)):
        if not path.is_file():
            sys.exit(f"Missing {label}: {path}")
    for label, path in (("GEN", args.gen_checkpoint), ("INTEN", args.inten_checkpoint)):
        if not path.is_file():
            sys.exit(f"Missing {label} checkpoint: {path}")
    if not args.predict_script.is_file():
        sys.exit(f"Missing predict script: {args.predict_script}")

    candidates, skipped_candidates = load_candidate_pool(args.candidates, args.smiles_column)
    if not candidates:
        sys.exit("No valid candidate SMILES found")
    if skipped_candidates:
        print(f"Skipped {len(skipped_candidates)} invalid or duplicate candidate rows")

    mgf_meta, truth = base.read_mgf_metadata_and_truth(args.mgf)
    identifiers = list(mgf_meta.keys())
    if args.identifier:
        wanted = set(args.identifier)
        identifiers = [ident for ident in identifiers if ident in wanted]
    if not identifiers:
        sys.exit("No matching MGF identifiers to score")

    print(f"MGF spectra: {len(identifiers)}; candidate pool: {len(candidates)}")

    if args.skip_predict:
        print("--skip-predict: validated inputs; no output written.")
        return

    ranking_rows: List[Dict[str, object]] = []

    with tempfile.TemporaryDirectory(prefix="iceberg_pool_rank_") as tmp:
        tmp_dir = Path(tmp)
        predictor_csv = tmp_dir / "candidates_for_predictor.csv"
        with predictor_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["identifier", "candidate_smiles"])
            writer.writeheader()
            for ident in identifiers:
                for cand in candidates:
                    writer.writerow({"identifier": ident, "candidate_smiles": cand["smiles"]})

        labels_tsv = tmp_dir / "candidates_labels.tsv"
        base.write_candidates_labels(predictor_csv, mgf_meta, labels_tsv)

        preds_h5 = args.preds_h5 if args.preds_h5 else run_predictor(args, labels_tsv, tmp_dir / "preds")
        if not preds_h5.is_file():
            sys.exit(f"Missing predictions HDF5: {preds_h5}")

        with h5py.File(preds_h5, "r") as h5:
            num_bins = int(h5.attrs.get("num_bins", 15000))
            upper_limit = float(h5.attrs.get("upper_limit", 1500.0))

            for ident in identifiers:
                peaks = truth.get(ident)
                if not peaks:
                    print(f"Skipping {ident}: no experimental peaks in MGF")
                    continue

                truth_vec = base._bin_spectra_local([peaks], num_bins=num_bins, upper_limit=upper_limit)[0]
                sims: List[Tuple[float, int]] = []
                for idx, _cand in enumerate(candidates):
                    spec = f"{ident}__cand{idx:02d}"
                    sims.append((score_candidate(h5, spec, truth_vec), idx))

                sims.sort(key=lambda item: (-item[0], item[1]))
                for rank, (sim, cand_idx) in enumerate(sims, start=1):
                    cand = candidates[cand_idx]
                    ranking_rows.append(
                        {
                            "identifier": ident,
                            "rank": rank,
                            "similarity": sim,
                            "smiles": cand["smiles"],
                            "canonical_smiles": cand["canonical_smiles"],
                            "candidate_pool_index": cand["candidate_pool_index"],
                        }
                    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "pool_candidate_ranking.csv"
    write_dicts(out_csv, ranking_rows)
    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()
