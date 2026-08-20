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

CLI entry for retrieval eval; implementation in ``src/foriceberg/ms_pred/retrieval/``.

See repo ``README.md``.
"""
from __future__ import annotations

import sys
from pathlib import Path

from foriceberg_paths import pythonpath_src

_GITHUB_SRC = Path(pythonpath_src())
if str(_GITHUB_SRC) not in sys.path:
    sys.path.insert(0, str(_GITHUB_SRC))

from ms_pred.retrieval.run_iceberg_retrieval_eval_evaltest import main  # noqa: E402

if __name__ == "__main__":
    main()
