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

Train gen + inten models on an arbitrary prepared FMS2Lib ICEBERG dataset.

Mirrors the F_ICEBERG retrain hyper-parameters (generation then intensity)
and accepts ``--dataset-name`` so smoke examples can use
``F_ICEBERG_train100`` without touching the full ``F_ICEBERG_dataset``.

Prerequisites under ``data/foriceberg/spec_datasets/<dataset-name>/``:
  labels.tsv, splits/split_1.tsv, magma_outputs/magma_tree.hdf5,
  and preferably subformulae/no_subform.hdf5 (falls back to spec_files.hdf5).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from foriceberg_paths import data_spec_datasets, github_root, pythonpath_src, results_dir, script_dir


def subprocess_python() -> str:
    raw = (os.environ.get("ICEBERG_PYTHON") or "").strip()
    if raw:
        cand = Path(raw).expanduser().resolve()
        if cand.is_file():
            return str(cand)
    return sys.executable


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ICEBERG gen/inten on a prepared dataset.")
    p.add_argument(
        "--dataset-name",
        default="F_ICEBERG_train100",
        help="Dataset folder under data/foriceberg/spec_datasets/",
    )
    p.add_argument(
        "--run-name",
        default=None,
        help="Output run folder name (default: <dataset>_rnd1_docx)",
    )
    p.add_argument("--split-name", default="split_1.tsv")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument(
        "--max-nodes",
        type=int,
        default=100,
        help="Max nodes for predict_gen (docx uses 100)",
    )
    p.add_argument(
        "--skip-existing-check",
        action="store_true",
        help="Allow continuing if output dirs already exist (dangerous).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = github_root()
    add_script = script_dir() / "add_dag_intens.py"
    if not add_script.is_file():
        sys.exit(f"ERROR: missing {add_script}")

    os.chdir(root)
    env = os.environ.copy()
    env["PYTHONPATH"] = pythonpath_src()
    env.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    dataset_root = data_spec_datasets(args.dataset_name)
    labels_path = dataset_root / "labels.tsv"
    split_path = dataset_root / "splits" / args.split_name
    magma_tree = dataset_root / "magma_outputs" / "magma_tree.hdf5"
    legacy_subform = dataset_root / "subformulae" / "no_subform.hdf5"
    spec_files = dataset_root / "spec_files.hdf5"

    for req in (labels_path, split_path, magma_tree):
        if not req.is_file():
            sys.exit(
                f"ERROR: missing {req}\n"
                "Prepare the dataset first (MGF -> labels/spec_files -> subformulae -> MAGMA)."
            )

    run_name = args.run_name or f"{args.dataset_name}_rnd1_docx"
    results_dir().mkdir(parents=True, exist_ok=True)
    gen_dir = results_dir() / f"dag_{args.dataset_name}" / run_name
    inten_dir = results_dir() / f"dag_inten_{args.dataset_name}" / run_name
    gen_preds_dir = gen_dir / "preds_train_100"
    gen_inten_h5 = gen_dir / "preds_train_100_inten.hdf5"

    if not args.skip_existing_check and (gen_dir.exists() or inten_dir.exists()):
        print("ERROR: output directory already exists. Remove results first if you want to rerun.")
        print(f"  {gen_dir}")
        print(f"  {inten_dir}")
        sys.exit(1)

    py = subprocess_python()

    def run(cmd: list[str]) -> None:
        print("+", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=str(root), env=env, check=True)

    print(f"[train] cwd={root}")
    print(f"[train] python={py}")
    print(f"[train] dataset={dataset_root}")
    print(f"[train] PYTHONPATH={env['PYTHONPATH']}")

    gen_cmd = [
        py,
        "-m",
        "ms_pred.dag_pred.train_gen",
        "--dataset-name",
        args.dataset_name,
        "--dataset-labels",
        "labels.tsv",
        "--split-name",
        args.split_name,
        "--save-dir",
        str(gen_dir.relative_to(root)),
        "--magma-folder",
        "magma_outputs",
        "--seed",
        str(args.seed),
        "--num-workers",
        str(args.num_workers),
        "--batch-size",
        str(args.batch_size),
        "--max-epochs",
        str(args.max_epochs),
        "--learning-rate",
        "0.000996",
        "--lr-decay-rate",
        "0.7214",
        "--dropout",
        "0.2",
        "--mpnn-type",
        "GGNN",
        "--pe-embed-k",
        "14",
        "--pool-op",
        "avg",
        "--set-layers",
        "2",
        "--hidden-size",
        "512",
        "--weight-decay",
        "0.0",
        "--layers",
        "3",
        "--root-encode",
        "gnn",
        "--encode-forms",
        "--embed-adduct",
        "--embed-collision",
        "--embed-elem-group",
        "--add-hs",
    ]
    if args.gpu:
        gen_cmd.append("--gpu")
    run(gen_cmd)

    gen_ckpt = gen_dir / "version_0" / "best.ckpt"
    if not gen_ckpt.is_file():
        sys.exit(f"ERROR: missing checkpoint {gen_ckpt}")

    pred_cmd = [
        py,
        "-m",
        "ms_pred.dag_pred.predict_gen",
        "--checkpoint-pth",
        str(gen_ckpt),
        "--dataset-name",
        args.dataset_name,
        "--dataset-labels",
        "labels.tsv",
        "--split-name",
        args.split_name,
        "--save-dir",
        str(gen_preds_dir.relative_to(root)),
        "--batch-size",
        "8",
        "--num-workers",
        str(args.num_workers),
        "--threshold",
        "0.0",
        "--max-nodes",
        str(args.max_nodes),
        "--num-decoys",
        "0",
    ]
    if args.gpu:
        pred_cmd.append("--gpu")
    run(pred_cmd)

    tree_preds = gen_preds_dir / "tree_preds.hdf5"
    if not tree_preds.is_file():
        sys.exit(f"ERROR: missing {tree_preds}")

    true_dag = legacy_subform if legacy_subform.is_file() else spec_files
    if not true_dag.is_file():
        sys.exit(f"ERROR: missing intensity source ({legacy_subform} or {spec_files})")

    add_cmd = [
        py,
        str(add_script),
        "--pred-dag-folder",
        str(tree_preds),
        "--true-dag-folder",
        str(true_dag),
        "--out-dag-folder",
        str(gen_inten_h5),
        "--num-workers",
        str(args.num_workers),
        "--add-raw",
    ]
    print("+", " ".join(add_cmd), flush=True)
    r = subprocess.run(add_cmd, cwd=str(root), env=env)
    if r.returncode != 0 and true_dag != spec_files and spec_files.is_file():
        print("[train] add_dag_intens (subform) failed; retry with spec_files.hdf5", flush=True)
        subprocess.run(
            [
                py,
                str(add_script),
                "--pred-dag-folder",
                str(tree_preds),
                "--true-dag-folder",
                str(spec_files),
                "--out-dag-folder",
                str(gen_inten_h5),
                "--num-workers",
                str(args.num_workers),
                "--add-raw",
            ],
            cwd=str(root),
            env=env,
            check=True,
        )
    elif r.returncode != 0:
        sys.exit(r.returncode)

    if not gen_inten_h5.is_file():
        sys.exit(f"ERROR: missing {gen_inten_h5}")

    inten_cmd = [
        py,
        "-m",
        "ms_pred.dag_pred.train_inten",
        "--dataset-name",
        args.dataset_name,
        "--dataset-labels",
        "labels.tsv",
        "--split-name",
        args.split_name,
        "--magma-dag-folder",
        str(gen_inten_h5.relative_to(root)),
        "--save-dir",
        str(inten_dir.relative_to(root)),
        "--seed",
        str(args.seed),
        "--num-workers",
        str(args.num_workers),
        "--batch-size",
        str(args.batch_size),
        "--max-epochs",
        str(args.max_epochs),
        "--learning-rate",
        "0.000736",
        "--lr-decay-rate",
        "0.825",
        "--dropout",
        "0.1",
        "--mpnn-type",
        "GGNN",
        "--pe-embed-k",
        "0",
        "--pool-op",
        "avg",
        "--hidden-size",
        "256",
        "--weight-decay",
        "1e-7",
        "--set-layers",
        "0",
        "--frag-set-layers",
        "3",
        "--mlp-layers",
        "1",
        "--gnn-layers",
        "4",
        "--loss-fn",
        "entropy",
        "--root-encode",
        "gnn",
        "--include-unshifted-mz",
        "--binned-targs",
        "--embed-adduct",
        "--embed-collision",
        "--embed-elem-group",
        "--encode-forms",
        "--add-hs",
        "--sk-tau",
        "0.05",
        "--ppm-tol",
        "20",
    ]
    if args.gpu:
        inten_cmd.append("--gpu")
    run(inten_cmd)

    print("[train] DONE")
    print(f"  Gen:   {gen_dir / 'version_0' / 'best.ckpt'}")
    print(f"  Inten: {inten_dir / 'version_0' / 'best.ckpt'}")


if __name__ == "__main__":
    main()
