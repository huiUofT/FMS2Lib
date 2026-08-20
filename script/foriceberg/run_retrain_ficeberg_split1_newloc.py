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

F_ICEBERG split_1 retrain (legacy_merge).

Run directly or via ``bash sh/foriceberg/retrain_ficeberg_split1_newloc.sh``. See repo ``README.md``.

- Cwd: FMS2Lib repo root; ``PYTHONPATH``: ``src/foriceberg``.
- Dataset: ``data/foriceberg/spec_datasets/F_ICEBERG_dataset/`` (labels, splits, magma, subformulae).
- Outputs: ``script/foriceberg/results/dag_F_ICEBERG_retrain/`` and ``dag_inten_F_ICEBERG_retrain/``.

Env: ``ICEBERG_GITHUB_ROOT`` (repo root), ``ICEBERG_PYTHON``, ``ICEBERG_STRICT_GITHUB=1``.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def subprocess_python() -> str:
    """Prefer ICEBERG_PYTHON so train steps match conda/venv even if the driver was mis-invoked."""
    raw = (os.environ.get("ICEBERG_PYTHON") or "").strip()
    if raw:
        cand = Path(raw).expanduser().resolve()
        if cand.is_file():
            return str(cand)
    return sys.executable


from foriceberg_paths import (  # noqa: E402
    data_spec_datasets,
    github_root,
    pythonpath_src,
    results_dir,
    script_dir,
)


def _github_root() -> Path:
    return github_root()


def _resolved_under_tree(path: Path, root: Path) -> bool:
    """True if path exists and resolve() stays under repo root (strict mode)."""
    try:
        if not path.exists():
            return False
        r = root.resolve()
        q = path.resolve()
    except OSError:
        return False
    try:
        q.relative_to(r)
        return True
    except ValueError:
        return False


def main() -> None:
    root = _github_root()
    add_script = script_dir() / "add_dag_intens.py"
    if not add_script.is_file():
        print(f"ERROR: missing {add_script}", file=sys.stderr)
        sys.exit(2)

    os.chdir(root)
    env = os.environ.copy()
    strict = os.environ.get("ICEBERG_STRICT_GITHUB", "").strip() in ("1", "true", "yes", "on")
    # Only src/foriceberg for ms_pred; do not append inherited PYTHONPATH.
    env["PYTHONPATH"] = pythonpath_src()
    env.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    dataset_root = data_spec_datasets("F_ICEBERG_dataset")
    spec_files = dataset_root / "spec_files.hdf5"
    labels_path = dataset_root / "labels.tsv"
    split_path = dataset_root / "splits/split_1.tsv"
    magma_tree = dataset_root / "magma_outputs/magma_tree.hdf5"
    legacy_subform = dataset_root / "subformulae/no_subform.hdf5"

    for req in (spec_files, labels_path, split_path, magma_tree):
        if not req.is_file():
            print(
                f"ERROR: missing {req}\n"
                f"Prepare F_ICEBERG_dataset under {data_spec_datasets()} "
                f"(see README.md at the FMS2Lib repo root, ICEBERG section).",
                file=sys.stderr,
            )
            sys.exit(1)

    if strict:
        to_check = [
            ("spec_files", spec_files),
            ("labels", labels_path),
            ("split", split_path),
            ("magma_tree", magma_tree),
        ]
        if legacy_subform.is_file():
            to_check.append(("no_subform", legacy_subform))
        for label, req in to_check:
            if not _resolved_under_tree(req, root):
                print(
                    f"ERROR: ICEBERG_STRICT_GITHUB=1 but {label} is not confined to repo root after resolve():\n"
                    f"  {req}\n"
                    f"Copy or bind-mount data under {data_spec_datasets()} instead of symlinks to other trees.",
                    file=sys.stderr,
                )
                sys.exit(1)
    run_name = "split_1_rnd1_retrain_legacy_merge"
    results_dir().mkdir(parents=True, exist_ok=True)
    gen_dir = results_dir() / "dag_F_ICEBERG_retrain" / run_name
    inten_dir = results_dir() / "dag_inten_F_ICEBERG_retrain" / run_name
    gen_preds_dir = gen_dir / "preds_train_100"
    gen_inten_h5 = gen_dir / "preds_train_100_inten.hdf5"

    if gen_dir.exists() or inten_dir.exists():
        print("ERROR: output directory already exists. Remove results first if you want to rerun.", file=sys.stderr)
        print(f"  {gen_dir}", file=sys.stderr)
        print(f"  {inten_dir}", file=sys.stderr)
        sys.exit(1)

    py = subprocess_python()

    def run(cmd: list[str]) -> None:
        print("+", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=str(root), env=env, check=True)

    print(f"[retrain] cwd={root}")
    print(f"[retrain] python={py}")
    print(f"[retrain] PYTHONPATH={env['PYTHONPATH']}")
    if strict:
        print("[retrain] ICEBERG_STRICT_GITHUB=1 (data + imports confined to repo root)", flush=True)

    run(
        [
            py,
            "-m",
            "ms_pred.dag_pred.train_gen",
            "--dataset-name",
            "F_ICEBERG_dataset",
            "--dataset-labels",
            "labels.tsv",
            "--split-name",
            "split_1.tsv",
            "--save-dir",
            str(gen_dir.relative_to(root)),
            "--magma-folder",
            "magma_outputs",
            "--seed",
            "1",
            "--gpu",
            "--num-workers",
            "8",
            "--batch-size",
            "32",
            "--max-epochs",
            "200",
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
    )

    gen_ckpt = gen_dir / "version_0/best.ckpt"
    if not gen_ckpt.is_file():
        print(f"ERROR: missing checkpoint {gen_ckpt}", file=sys.stderr)
        sys.exit(1)

    run(
        [
            py,
            "-m",
            "ms_pred.dag_pred.predict_gen",
            "--checkpoint-pth",
            str(gen_ckpt),
            "--dataset-name",
            "F_ICEBERG_dataset",
            "--dataset-labels",
            "labels.tsv",
            "--split-name",
            "split_1.tsv",
            "--save-dir",
            str(gen_preds_dir.relative_to(root)),
            "--batch-size",
            "8",
            "--num-workers",
            "8",
            "--threshold",
            "0.0",
            "--max-nodes",
            "100",
            "--num-decoys",
            "0",
            "--gpu",
        ]
    )

    tree_preds = gen_preds_dir / "tree_preds.hdf5"
    if not tree_preds.is_file():
        print(f"ERROR: missing {tree_preds}", file=sys.stderr)
        sys.exit(1)

    add_cmd = [
        py,
        str(add_script),
        "--pred-dag-folder",
        str(tree_preds),
        "--true-dag-folder",
        str(legacy_subform),
        "--out-dag-folder",
        str(gen_inten_h5),
        "--num-workers",
        "8",
        "--add-raw",
    ]
    print("+", " ".join(add_cmd), flush=True)
    r = subprocess.run(add_cmd, cwd=str(root), env=env)
    if r.returncode != 0:
        print("[retrain] add_dag_intens (subform) failed; retry with spec_files.hdf5", flush=True)
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
                "8",
                "--add-raw",
            ],
            cwd=str(root),
            env=env,
            check=True,
        )

    if not gen_inten_h5.is_file():
        print(f"ERROR: missing {gen_inten_h5}", file=sys.stderr)
        sys.exit(1)

    run(
        [
            py,
            "-m",
            "ms_pred.dag_pred.train_inten",
            "--dataset-name",
            "F_ICEBERG_dataset",
            "--dataset-labels",
            "labels.tsv",
            "--split-name",
            "split_1.tsv",
            "--magma-dag-folder",
            str(gen_inten_h5.relative_to(root)),
            "--save-dir",
            str(inten_dir.relative_to(root)),
            "--seed",
            "1",
            "--gpu",
            "--num-workers",
            "8",
            "--batch-size",
            "32",
            "--max-epochs",
            "200",
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
    )

    print("[retrain] DONE")
    print(f"  Gen:   {gen_dir / 'version_0/best.ckpt'}")
    print(f"  Inten: {inten_dir / 'version_0/best.ckpt'}")


if __name__ == "__main__":
    main()
