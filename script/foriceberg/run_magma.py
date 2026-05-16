#!/usr/bin/env python3
"""
Portable entry for MAGMA fragmentation trees.

Implementation: ``src/foriceberg/ms_pred/magma/run_magma.py``.
"""
from __future__ import annotations

import os
import subprocess
import sys

from foriceberg_paths import github_root, pythonpath_src, src_dir


def main() -> None:
    root = github_root()
    module_main = src_dir() / "ms_pred" / "magma" / "run_magma.py"
    if not module_main.is_file():
        print(f"ERROR: missing {module_main}", file=sys.stderr)
        sys.exit(2)

    env = os.environ.copy()
    env["PYTHONPATH"] = pythonpath_src() if not (env.get("PYTHONPATH") or "").strip() else (
        pythonpath_src() + os.pathsep + env["PYTHONPATH"]
    )

    argv = [sys.executable, "-m", "ms_pred.magma.run_magma", *sys.argv[1:]]
    print("+", " ".join(argv), flush=True)
    raise SystemExit(subprocess.run(argv, cwd=str(root), env=env).returncode)


if __name__ == "__main__":
    main()
