#!/usr/bin/env python3
"""
Unified CLI for ICEBERG workflows under ``script/foriceberg/``.

See ``README.md`` at the FMS2Lib repo root (ICEBERG / foriceberg section).
"""
from __future__ import annotations

import os
import subprocess
import sys

from foriceberg_paths import github_root, pythonpath_src, repo_root, script_dir

COMMANDS: dict[str, tuple[str, str]] = {
    "process-dataset": (
        "process_new_dataset_from_mgf_with_parentmass.py",
        "MGF -> labels.tsv + spec_files.hdf5 + split_<seed>.tsv",
    ),
    "magma": (
        "run_magma.py",
        "spec_files.hdf5 + labels.tsv -> magma_outputs/magma_tree.hdf5",
    ),
    "end2end": (
        "run_end2end_from_mgf_susdat_onepass.py",
        "MGF + SusDat CSV -> candidates -> predict -> Top-1 eval",
    ),
    "smiles2mgf": (
        "smiles_to_predicted_mgf.py",
        "SMILES -> predicted MS/MS MGF",
    ),
    "predict-eval": (
        "run_predict_and_evaluate_testset.py",
        "labels TSV -> preds.hdf5 -> cosine evaluation",
    ),
    "retrieval-eval": (
        "run_iceberg_retrieval_eval_evaltest.py",
        "candidate CSV + MGF -> dist-bin retrieval eval",
    ),
    "blind-retrieval": (
        "blind_retrieval_from_pred_formula.py",
        "predicted formula + MGF -> blind candidate retrieval",
    ),
    "pubchem-top100": (
        "build_pubchem_top100_by_ecfp4_from_mgf.py",
        "MGF + PubChem HDF5 -> ECFP4-ranked candidate CSV",
    ),
    "retrain": (
        "run_retrain_ficeberg_split1_newloc.py",
        "F_ICEBERG split_1 legacy_merge retrain (gen + inten)",
    ),
}


def _usage() -> str:
    lines = [
        "usage: python script/foriceberg/run.py <command> [args...]",
        "",
        "commands:",
    ]
    width = max(len(k) for k in COMMANDS)
    for name, (_, desc) in sorted(COMMANDS.items()):
        lines.append(f"  {name.ljust(width)}  {desc}")
    lines.append("")
    lines.append(f"FMS2Lib root: {repo_root()}")
    lines.append("documentation: README.md (repo root, ICEBERG section)")
    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(_usage())
        return

    cmd = sys.argv[1]
    if cmd not in COMMANDS:
        print(f"unknown command: {cmd}\n", file=sys.stderr)
        print(_usage(), file=sys.stderr)
        sys.exit(2)

    script_name, _ = COMMANDS[cmd]
    script_path = script_dir() / script_name
    if not script_path.is_file():
        print(f"ERROR: missing {script_path}", file=sys.stderr)
        sys.exit(2)

    root = github_root()
    env = os.environ.copy()
    env["PYTHONPATH"] = pythonpath_src() if not (env.get("PYTHONPATH") or "").strip() else (
        pythonpath_src() + os.pathsep + env["PYTHONPATH"]
    )

    argv = [sys.executable, str(script_path), *sys.argv[2:]]
    print("+", " ".join(argv), flush=True)
    raise SystemExit(subprocess.run(argv, cwd=str(root), env=env).returncode)


if __name__ == "__main__":
    main()
