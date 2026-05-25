#!/usr/bin/env python3
"""
End-to-end: MGF + per-spectrum formulas -> Msbuddy feature matrix (joblibs) ->
LightGBM LambdaRank train + held-out Test Top-1 metrics -> saved ranker model.

Stages (same semantics as calling generate_training_data.py then train_ranker.py):

1) Feature generation: writes under --output-dir
     X_arr.joblib, y_arr.joblib, group_arr.joblib, formula_strs.joblib, summary.log
2) Ranker train + eval: writes under --output-dir/trained_models/ (unless --model-dir)
     model_ms1_ms2_ranker.joblib, train_ms1_ms2_summary.txt

This is supervised training/eval only: formulas must align with spectra (TXT, one formula
per line, same order as MGF blocks). For raw MGF formula discovery without supervision,
use MsBuddy/other workflows — not covered here.

cwd for subprocess runs: inferred project root (--repo-root) so relative paths in args resolve.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def infer_repo_root(script_dir: Path) -> Path:
    """
    Default subprocess cwd: repository root containing ``src/msbuddy`` and ``script``.
    Override with ``--repo-root`` if wrong.
    """
    d = script_dir.resolve()
    for anc in d.parents:
        if (anc / "src" / "msbuddy").is_dir() and (anc / "script").is_dir():
            return anc
    return d.parent.parent.parent


def run_step(cmd: list[str], cwd: Path) -> None:
    print("\n" + "=" * 72)
    print(" ", cmd[0])
    print(" ".join(cmd[1:]))
    print("=" * 72, flush=True)
    r = subprocess.run(cmd, cwd=str(cwd))
    if r.returncode != 0:
        sys.exit(r.returncode)


def main(argv: list[str] | None = None) -> int:
    script_dir = Path(__file__).resolve().parent

    default_root = infer_repo_root(script_dir)

    ap = argparse.ArgumentParser(
        description="One script: MGF + formulas -> feature joblibs -> train ranker + test metrics.",
    )
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Working directory for subprocesses (default: inferred repository root).",
    )
    ap.add_argument("--mgf", type=str, required=True, help="Input MGF path")
    ap.add_argument(
        "--formulas",
        type=str,
        required=True,
        help="TXT: one precursor formula per line, same order as MGF spectra",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output folder for step-1 artifacts; ranker reads from here (--data-dir)",
    )
    ap.add_argument(
        "--skip-generate",
        action="store_true",
        help="Only run ranker (joblibs already in --output-dir)",
    )
    ap.add_argument(
        "--skip-train",
        action="store_true",
        help="Only run feature generation",
    )

    # generate_training_data
    ap.add_argument("--ms1-tol", type=float, default=2.0)
    ap.add_argument("--ms2-tol", type=float, default=5.0)
    ap.add_argument("--random-seed", type=int, default=24)
    ap.add_argument("--n-cpu", type=int, default=-1)
    ap.add_argument("--batch-size", type=int, default=20000)
    ap.add_argument("--max-candidates-per-spectrum", type=int, default=None)

    # train_ranker
    ap.add_argument("--model-dir", type=str, default=None)
    ap.add_argument("--val-size", type=float, default=0.1)
    ap.add_argument("--test-size", type=float, default=0.1)
    ap.add_argument("--no-mmap", action="store_true")
    ap.add_argument("--lgb-jobs", type=int, default=1)

    args = ap.parse_args(argv)

    cwd = (args.repo_root.expanduser().resolve() if args.repo_root else default_root.resolve())
    py = sys.executable
    gen = script_dir / "generate_training_data.py"
    rank = script_dir / "train_ranker.py"

    for need in (gen, rank):
        if not need.is_file():
            print(f"ERROR: missing sibling script {need}")
            return 2

    mgf = str(Path(args.mgf).expanduser())
    formulas = str(Path(args.formulas).expanduser())
    out_dir = str(Path(args.output_dir).expanduser())

    if not args.skip_generate:
        gen_cmd = [
            py,
            str(gen),
            "--mgf",
            mgf,
            "--formulas",
            formulas,
            "--output-dir",
            out_dir,
            "--ms1-tol",
            str(args.ms1_tol),
            "--ms2-tol",
            str(args.ms2_tol),
            "--random-seed",
            str(args.random_seed),
            "--n-cpu",
            str(args.n_cpu),
            "--batch-size",
            str(args.batch_size),
        ]
        if args.max_candidates_per_spectrum is not None:
            gen_cmd += [
                "--max-candidates-per-spectrum",
                str(args.max_candidates_per_spectrum),
            ]
        run_step(gen_cmd, cwd)

    if not args.skip_train:
        rank_cmd = [
            py,
            str(rank),
            "--data-dir",
            out_dir,
            "--random-seed",
            str(args.random_seed),
            "--val-size",
            str(args.val_size),
            "--test-size",
            str(args.test_size),
            "--lgb-jobs",
            str(args.lgb_jobs),
        ]
        if args.model_dir:
            rank_cmd.extend(["--model-dir", str(Path(args.model_dir).expanduser())])
        if args.no_mmap:
            rank_cmd.append("--no-mmap")
        run_step(rank_cmd, cwd)

    outp = Path(out_dir).expanduser().resolve()
    mdir = Path(args.model_dir).expanduser().resolve() if args.model_dir else (outp / "trained_models")

    print("\n" + "=" * 72)
    print("Done.")
    print("Artifacts:")
    print(f"  feature dir:     {outp}")
    print(f"    X_arr.joblib | y_arr.joblib | group_arr.joblib | formula_strs.joblib")
    print(f"    summary.log")
    print(f"  ranker dir:       {mdir}")
    print(f"    model_ms1_ms2_ranker.joblib")
    print(f"    train_ms1_ms2_summary.txt")
    print(f" subprocess cwd was: {cwd}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
