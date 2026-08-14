#!/usr/bin/env python3
"""
Portable entry for MAGMA fragmentation trees.

Implementation: ``src/foriceberg/ms_pred/magma/run_magma.py``.

After a successful run, ensures ``magma_tree.hdf5`` exists (symlink to
``magma_tree_new.hdf5``) because training reads the former name.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from foriceberg_paths import github_root, pythonpath_src, src_dir


def _ensure_magma_tree_alias(output_dir: Path) -> None:
    new_tree = output_dir / "magma_tree_new.hdf5"
    alias = output_dir / "magma_tree.hdf5"
    if not new_tree.is_file():
        return
    if alias.exists() or alias.is_symlink():
        alias.unlink()
    try:
        alias.symlink_to(new_tree.name)
        print(f"[magma] linked {alias.name} -> {new_tree.name}", flush=True)
    except OSError:
        import shutil

        shutil.copy2(new_tree, alias)
        print(f"[magma] copied {new_tree.name} -> {alias.name}", flush=True)

    new_tsv = output_dir / "magma_tsv_new.hdf5"
    alias_tsv = output_dir / "magma_tsv.hdf5"
    if new_tsv.is_file():
        if alias_tsv.exists() or alias_tsv.is_symlink():
            alias_tsv.unlink()
        try:
            alias_tsv.symlink_to(new_tsv.name)
        except OSError:
            import shutil

            shutil.copy2(new_tsv, alias_tsv)


def main() -> None:
    root = github_root()
    module_main = src_dir() / "ms_pred" / "magma" / "run_magma.py"
    if not module_main.is_file():
        print(f"ERROR: missing {module_main}", file=sys.stderr)
        sys.exit(2)

    # Peek --output-dir without consuming argv for the subprocess.
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--output-dir", default=None)
    known, _ = peek.parse_known_args(sys.argv[1:])

    env = os.environ.copy()
    env["PYTHONPATH"] = pythonpath_src() if not (env.get("PYTHONPATH") or "").strip() else (
        pythonpath_src() + os.pathsep + env["PYTHONPATH"]
    )

    argv = [sys.executable, "-m", "ms_pred.magma.run_magma", *sys.argv[1:]]
    print("+", " ".join(argv), flush=True)
    code = subprocess.run(argv, cwd=str(root), env=env).returncode
    if code == 0 and known.output_dir:
        _ensure_magma_tree_alias(Path(known.output_dir))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
