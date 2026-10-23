#!/usr/bin/env python3
"""This code was adapted from the ICEBERG implementation developed by
Goldman et al.

Original repository:
https://github.com/samgoldman97/ms-pred

The ICEBERG model was retrained using an in-house MS/MS dataset of
low-fluorinated compounds. Modifications include dataset preparation,
data splitting, model training, and performance evaluation.

Modifications by: Tao Yang
Date: 20260819

The original copyright notice and license terms are retained.

Predict test labels then evaluate cosine similarity.

Runs ``src/foriceberg/ms_pred/dag_pred/predict_smis_trained_inten.py`` then
``evaluate_test_set.py``. Default checkpoints: ``mode/foriceberg/generation/best.ckpt``,
``mode/foriceberg/intensity/best.ckpt``. See repo ``README.md``.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from foriceberg_paths import (
    github_root,
    mode_dir,
    pythonpath_src,
    results_dir,
    script_dir,
    src_dir,
)


def _pythonpath_github_src_first(root: Path, base: dict[str, str]) -> str:
    """``src/foriceberg`` first so ``ms_pred`` is vendored; keep rest of PYTHONPATH (RDKit modules, etc.)."""
    prefix = pythonpath_src()
    raw = (base.get("PYTHONPATH") or "").strip()
    parts = [p.strip() for p in raw.split(os.pathsep) if p.strip()]
    out: list[str] = []
    pfx = Path(prefix)
    for p in parts:
        try:
            if Path(p).expanduser().resolve() == pfx:
                continue
        except OSError:
            pass
        out.append(p)
    return prefix if not out else prefix + os.pathsep + os.pathsep.join(out)


def _resolve_repo_path(root: Path, path_str: str) -> Path:
    p = Path(path_str)
    return (p if p.is_absolute() else root / p).resolve()


def main() -> None:
    p = argparse.ArgumentParser(description="Predict (joint inten) then evaluate cosine on test labels.")
    p.add_argument(
        "--gen-checkpoint",
        default=None,
        type=str,
        help="Gen .ckpt (default: mode/foriceberg/generation/best.ckpt under github root if present)",
    )
    p.add_argument(
        "--inten-checkpoint",
        default=None,
        type=str,
        help="Inten .ckpt (default: mode/foriceberg/intensity/best.ckpt under github root if present)",
    )
    p.add_argument(
        "--dataset-labels",
        required=True,
        type=str,
        help="Labels TSV for rows to predict (e.g. testset_split_1_labels.tsv)",
    )
    p.add_argument("--save-dir", required=True, type=str, help="Predictor output dir (will contain preds.hdf5)")
    p.add_argument("--output-dir", required=True, type=str, help="Evaluator output dir (evaluation_results.tsv, etc.)")
    p.add_argument("--dataset-name", required=True, type=str, help="Dataset folder name under data/foriceberg/spec_datasets/")
    p.add_argument("--gpu", action="store_true")
    p.add_argument(
        "--no-adduct-shift",
        action="store_true",
        help="Disable adduct shift (default: pass --adduct-shift to predictor)",
    )
    p.add_argument(
        "--no-sparse-out",
        action="store_true",
        help="Disable sparse output (default: --sparse-out)",
    )
    p.add_argument("--sparse-k", type=int, default=100)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-nodes", type=int, default=5000)
    p.add_argument("--threshold", type=float, default=0.05)
    p.add_argument("--num-bins", type=int, default=15000)
    p.add_argument("--upper-limit", type=int, default=1500)
    p.add_argument("--skip-predict", action="store_true", help="Only run evaluate (expects preds.hdf5 in --save-dir)")
    p.add_argument("--skip-eval", action="store_true", help="Only run predict")
    args = p.parse_args()

    root = github_root()

    if not args.skip_predict:
        mode_gen = mode_dir() / "generation" / "best.ckpt"
        mode_inten = mode_dir() / "intensity" / "best.ckpt"
        if not args.gen_checkpoint and mode_gen.is_file():
            args.gen_checkpoint = "mode/foriceberg/generation/best.ckpt"
        if not args.inten_checkpoint and mode_inten.is_file():
            args.inten_checkpoint = "mode/foriceberg/intensity/best.ckpt"
        if not args.gen_checkpoint or not args.inten_checkpoint:
            print(
                "ERROR: need --gen-checkpoint and --inten-checkpoint (unless --skip-predict).\n"
                f"Default bundle paths (if you ship weights here):\n"
                f"  {mode_gen}\n"
                f"  {mode_inten}",
                file=sys.stderr,
            )
            sys.exit(2)

    pred_py = src_dir() / "ms_pred" / "dag_pred" / "predict_smis_trained_inten.py"
    eval_py = script_dir() / "evaluate_test_set.py"
    for req in (pred_py, eval_py):
        if not req.is_file():
            print(f"ERROR: missing {req}", file=sys.stderr)
            sys.exit(2)

    if not args.skip_predict:
        for label, pth in (
            ("--gen-checkpoint", args.gen_checkpoint),
            ("--inten-checkpoint", args.inten_checkpoint),
        ):
            ck = _resolve_repo_path(root, pth)
            if not ck.is_file():
                print(
                    f"ERROR: {label} not found:\n  {ck}\n"
                    f"Relative paths are resolved from github root: {root}\n"
                    f"Shipped defaults: ``mode/foriceberg/generation/best.ckpt`` and ``mode/foriceberg/intensity/best.ckpt``.\n"
                    f"Or train (``bash sh/foriceberg/retrain_ficeberg_split1_newloc.sh``) and pass ``results/.../best.ckpt``.",
                    file=sys.stderr,
                )
                sys.exit(2)
        labels_p = _resolve_repo_path(root, args.dataset_labels)
        if not labels_p.is_file():
            print(
                f"ERROR: --dataset-labels not found:\n  {labels_p}",
                file=sys.stderr,
            )
            sys.exit(2)

    os.chdir(root)
    env = os.environ.copy()
    env["PYTHONPATH"] = _pythonpath_github_src_first(root, env)

    py = sys.executable

    def run(cmd: list[str]) -> None:
        print("+", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=str(root), env=env, check=True)

    print(f"[predict+eval] cwd={root}", flush=True)
    print(f"[predict+eval] PYTHONPATH={env['PYTHONPATH']}", flush=True)

    if not args.skip_predict:
        pred_cmd = [
            py,
            str(pred_py),
            "--gen-checkpoint",
            args.gen_checkpoint,
            "--inten-checkpoint",
            args.inten_checkpoint,
            "--dataset-labels",
            args.dataset_labels,
            "--save-dir",
            args.save_dir,
            "--num-workers",
            str(args.num_workers),
            "--batch-size",
            str(args.batch_size),
            "--max-nodes",
            str(args.max_nodes),
            "--threshold",
            str(args.threshold),
            "--sparse-k",
            str(args.sparse_k),
        ]
        if not args.no_sparse_out:
            pred_cmd.append("--sparse-out")
        if not args.no_adduct_shift:
            pred_cmd.append("--adduct-shift")
        if args.gpu:
            pred_cmd.append("--gpu")
        run(pred_cmd)

    sd = Path(args.save_dir)
    preds_h5 = (sd if sd.is_absolute() else root / sd) / "preds.hdf5"
    preds_h5 = preds_h5.resolve()
    if not preds_h5.is_file():
        print(f"ERROR: missing predictions {preds_h5}", file=sys.stderr)
        sys.exit(1)

    try:
        pred_arg = str(preds_h5.relative_to(root.resolve()))
    except ValueError:
        pred_arg = str(preds_h5)

    if not args.skip_eval:
        eval_cmd = [
            py,
            str(eval_py),
            "--pred-file",
            pred_arg,
            "--test-labels",
            args.dataset_labels,
            "--output-dir",
            args.output_dir,
            "--dataset-name",
            args.dataset_name,
            "--num-bins",
            str(args.num_bins),
            "--upper-limit",
            str(args.upper_limit),
        ]
        run(eval_cmd)

    print("[predict+eval] DONE", flush=True)
    print(f"  preds:   {preds_h5}", flush=True)
    out = Path(args.output_dir)
    out_disp = out if out.is_absolute() else root / out
    print(f"  eval out: {out_disp}", flush=True)


if __name__ == "__main__":
    main()
