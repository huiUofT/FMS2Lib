#!/usr/bin/env python3
"""
F_ICEBERG split_1 retrain (legacy_merge) — **portable under ``github/`` only**.

- Working directory: **this repository's ``github/`` root** (parent of ``script/``).
- ``PYTHONPATH``: **only** ``github/src`` (vendored ``ms_pred``).
- Dataset layout: ``github/data/spec_datasets/F_ICEBERG_dataset/`` (same as ICEBERG:
  ``labels.tsv``, ``splits/split_1.tsv``, ``magma_outputs/``, ``subformulae/``, …).
- Outputs: ``github/results/dag_F_ICEBERG_retrain/...`` and ``github/results/dag_inten_F_ICEBERG_retrain/...``.

Optional env (defaults shown):

- ``ICEBERG_GITHUB_ROOT``: override auto-detected ``github/`` root (absolute path).
- ``ICEBERG_PYTHON``: Python executable for training subprocesses (set by ``sh/retrain_ficeberg_split1_newloc.sh`` when conda is active, or set manually on HPC).
- ``ICEBERG_STRICT_GITHUB=1``: require every dataset file (after ``Path.resolve()``) to lie under ``github/``; subprocess ``PYTHONPATH`` is **only** ``github/src`` (no inherited paths).

For GitHub upload: ship this ``github/`` tree; place or symlink large HDF5/Tsv under
``github/data/spec_datasets/F_ICEBERG_dataset/`` (see ``github/data/README.md``).
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


def github_root() -> Path:
    if os.environ.get("ICEBERG_GITHUB_ROOT"):
        return Path(os.environ["ICEBERG_GITHUB_ROOT"]).expanduser().resolve()
    # github/script/this_file.py -> parent = github/
    return Path(__file__).resolve().parents[1]


def _resolved_under_tree(path: Path, root: Path) -> bool:
    """True if path exists and resolve() stays under root (symlinks outside github/ fail strict mode)."""
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
    root = github_root()
    add_script = root / "script" / "add_dag_intens.py"
    if not add_script.is_file():
        print(f"ERROR: missing {add_script}", file=sys.stderr)
        sys.exit(2)

    os.chdir(root)
    env = os.environ.copy()
    strict = os.environ.get("ICEBERG_STRICT_GITHUB", "").strip() in ("1", "true", "yes", "on")
    # Only ``github/src`` for ``ms_pred``; do not append inherited PYTHONPATH (avoids loading code from ICEBERG/src).
    env["PYTHONPATH"] = str(root / "src")
    env.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    dataset_root = root / "data/spec_datasets/F_ICEBERG_dataset"
    spec_files = dataset_root / "spec_files.hdf5"
    labels_path = dataset_root / "labels.tsv"
    split_path = dataset_root / "splits/split_1.tsv"
    magma_tree = dataset_root / "magma_outputs/magma_tree.hdf5"
    legacy_subform = dataset_root / "subformulae/no_subform.hdf5"

    for req in (spec_files, labels_path, split_path, magma_tree):
        if not req.is_file():
            print(
                f"ERROR: missing {req}\n"
                f"Prepare F_ICEBERG_dataset under {root / 'data/spec_datasets'} "
                f"(see github/data/README.md).",
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
                    f"ERROR: ICEBERG_STRICT_GITHUB=1 but {label} is not confined to github/ after resolve():\n"
                    f"  {req}\n"
                    f"Copy or bind-mount data under {root / 'data/spec_datasets'} instead of symlinks to other trees.",
                    file=sys.stderr,
                )
                sys.exit(1)
    run_name = "split_1_rnd1_retrain_legacy_merge"
    gen_dir = root / "results/dag_F_ICEBERG_retrain" / run_name
    inten_dir = root / "results/dag_inten_F_ICEBERG_retrain" / run_name
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
        print("[retrain] ICEBERG_STRICT_GITHUB=1 (data + imports confined to github/)", flush=True)

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
