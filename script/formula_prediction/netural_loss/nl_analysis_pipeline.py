#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline: subformula -> expand inter-fragment neutral losses -> per-compound NL frequency.

Example (from repo root):
  python final-database/py/nl_analysis_pipeline.py ^
    --mgf f-paper/FMS2lib_calibrated-F.mgf ^
    --formulas f-paper/FIg.1/origin-data/FMS2lib.csv ^
    --output-dir f-paper/FIg.1/origin-data/nl_analysis
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


def repo_root_from_here() -> Path:
    """Subprocess cwd: prefer ``f-paper``, else ``github`` root, else four parents (``py文件`` layout)."""
    p = Path(__file__).resolve()
    for anc in p.parents:
        if anc.name == "f-paper" and (anc / "github" / "src" / "msbuddy").is_dir():
            return anc
    for anc in p.parents:
        if (anc / "src" / "msbuddy").is_dir() and (anc / "script").is_dir():
            return anc
    return p.parent.parent.parent.parent


def default_msbuddy_dir() -> Path:
    root = repo_root_from_here()
    cand = root / "msbuddy-main"
    if cand.is_dir():
        return cand
    return cand


def meta_from_mgf_headers(mgf_path: Path, compound_key: str, out_csv: Path) -> bool:
    """If each spectrum block has FEATURE_ID and ``compound_key`` KEY=value lines, write a merge CSV."""
    rows: list[dict[str, str]] = []
    cur: dict[str, str] = {}
    in_block = False
    with open(mgf_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            raw = line.rstrip("\n")
            s = raw.strip()
            if s == "BEGIN IONS":
                in_block = True
                cur = {}
                continue
            if s == "END IONS":
                in_block = False
                fid = (cur.get("FEATURE_ID") or "").strip()
                comp = (cur.get(compound_key) or "").strip()
                if fid and comp:
                    rows.append({"FEATURE_ID": fid, compound_key: comp})
                continue
            if not in_block or "=" not in raw:
                continue
            k, _, v = raw.partition("=")
            k, v = k.strip(), v.strip()
            if k and v:
                cur[k] = v
    if not rows:
        return False
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["FEATURE_ID", compound_key]
    with open(out_csv, "w", newline="", encoding="utf-8") as wf:
        w = csv.DictWriter(wf, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({fn: r[fn] for fn in fieldnames})
    return True


def run_step(cmd: list[str], desc: str) -> None:
    print("\n" + "=" * 70)
    print(desc)
    print("=" * 70)
    print(" ", " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(repo_root_from_here()))
    if r.returncode != 0:
        sys.exit(r.returncode)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="NL pipeline: fragment CSV -> expanded NL -> compound frequency",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--mgf", required=True, help="Input MGF path")
    ap.add_argument(
        "--formulas",
        required=True,
        help="Formulas: TXT (one per line) or CSV with a formula column (align FEATURE_ID if present)",
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: <mgf parent>/nl_pipeline)",
    )
    ap.add_argument(
        "--msbuddy-dir",
        default=None,
        help="msbuddy-main root (default: <repo>/msbuddy-main)",
    )
    ap.add_argument("--ms2-tol", type=float, default=5.0, help="MS2 tolerance in ppm (default 5)")
    ap.add_argument(
        "--ms1-tol",
        type=float,
        default=2.0,
        help="MS1 tolerance in ppm (default 2; passed to subformula for logging/consistency)",
    )
    ap.add_argument("--n-cores", type=int, default=8, help="subformula worker count (default 8)")
    ap.add_argument(
        "--expand-jobs",
        type=int,
        default=6,
        help="Parallel jobs for calculate_inter_fragment_nl (default 6)",
    )
    ap.add_argument(
        "--meta-csv",
        default=None,
        help=(
            "Metadata CSV with FEATURE_ID and --compound-key column; "
            "default: same as --formulas if it is .csv. "
            "If omitted and --formulas is .txt, try to build from MGF KEY=value headers per spectrum."
        ),
    )
    ap.add_argument(
        "--compound-key",
        default="SMILES",
        help="Column name for compound (default SMILES)",
    )
    ap.add_argument(
        "--freq-output",
        default=None,
        help="Frequency table path (default: <output-dir>/nl_frequency_<compound-key>.csv)",
    )
    args = ap.parse_args()

    root = repo_root_from_here()
    py = sys.executable
    script_dir = Path(__file__).resolve().parent

    mgf = Path(args.mgf).expanduser()
    if not mgf.is_file():
        print(f"Error: MGF not found: {mgf}")
        sys.exit(1)

    formulas = Path(args.formulas).expanduser()
    if not formulas.is_file():
        print(f"Error: formulas file not found: {formulas}")
        sys.exit(1)

    if args.output_dir:
        out_dir = Path(args.output_dir).expanduser()
    else:
        out_dir = mgf.parent / "nl_pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)

    msbuddy = Path(args.msbuddy_dir).expanduser() if args.msbuddy_dir else default_msbuddy_dir()
    if not (msbuddy / "msbuddy").is_dir():
        print(f"Error: --msbuddy-dir is not a valid msbuddy package root: {msbuddy}")
        sys.exit(1)

    stem = mgf.stem
    fragments_csv = out_dir / f"{stem}_fragments.csv"
    expanded_csv = out_dir / f"{stem}_fragments_expanded_nl.csv"

    meta = args.meta_csv
    if meta is None and str(formulas).lower().endswith(".csv"):
        meta = str(formulas)
    meta_path = Path(meta).expanduser() if meta else None

    if args.freq_output:
        freq_out = Path(args.freq_output).expanduser()
    else:
        safe_key = "".join(c if c.isalnum() else "_" for c in args.compound_key)
        freq_out = out_dir / f"nl_frequency_{safe_key}.csv"

    run_step(
        [
            py,
            str(script_dir / "subformula.py"),
            "--mgf",
            str(mgf.resolve()),
            "--formulas",
            str(formulas.resolve()),
            "--msbuddy-dir",
            str(msbuddy.resolve()),
            "--output-dir",
            str(out_dir.resolve()),
            "--ms2-tol",
            str(args.ms2_tol),
            "--ms1-tol",
            str(args.ms1_tol),
            "--n-cores",
            str(args.n_cores),
        ],
        "Step 1/3: subformula (fragment table)",
    )

    if not fragments_csv.is_file():
        print(f"Error: expected file not created: {fragments_csv}")
        sys.exit(1)

    run_step(
        [
            py,
            str(script_dir / "calculate_inter_fragment_nl.py"),
            "-i",
            str(fragments_csv),
            "-o",
            str(expanded_csv),
            "-j",
            str(args.expand_jobs),
        ],
        "Step 2/3: calculate_inter_fragment_nl (expand neutral losses)",
    )

    if not expanded_csv.is_file():
        print(f"Error: expected file not created: {expanded_csv}")
        sys.exit(1)

    if not meta_path or not meta_path.is_file():
        built = out_dir / f"{stem}_nl_meta_from_mgf.csv"
        if meta_from_mgf_headers(mgf, args.compound_key, built):
            print(
                f"Note: no meta CSV for step 3; wrote {built.name} from MGF "
                f"(FEATURE_ID + {args.compound_key})."
            )
            meta_path = built

    if not meta_path or not meta_path.is_file():
        print(
            "Error: frequency step needs metadata: pass --meta-csv with FEATURE_ID + "
            f"{args.compound_key}, use --formulas as a .csv that contains those columns, "
            "or use an MGF where each BEGIN IONS block has FEATURE_ID= and "
            f"{args.compound_key}= lines."
        )
        sys.exit(1)

    run_step(
        [
            py,
            str(script_dir / "nl_frequency_by_compound.py"),
            "-i",
            str(expanded_csv),
            "-m",
            str(meta_path.resolve()),
            "--compound-key",
            args.compound_key,
            "-o",
            str(freq_out),
        ],
        "Step 3/3: nl_frequency_by_compound",
    )

    print("\n" + "=" * 70)
    print("Pipeline finished")
    print("=" * 70)
    print(f"  Fragments:  {fragments_csv}")
    print(f"  Expanded:   {expanded_csv}")
    print(f"  Frequency:  {freq_out}")


if __name__ == "__main__":
    main()
