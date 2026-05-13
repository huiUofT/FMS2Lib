#!/usr/bin/env python3
"""CLI entry: implementation in ms_pred.retrieval (github/src)."""
from __future__ import annotations

import sys
from pathlib import Path

_GITHUB_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_GITHUB_SRC) not in sys.path:
    sys.path.insert(0, str(_GITHUB_SRC))

from ms_pred.retrieval.run_iceberg_retrieval_eval_evaltest import main  # noqa: E402

if __name__ == "__main__":
    main()
