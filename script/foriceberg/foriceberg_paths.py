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

Path helpers for ICEBERG assets under FMS2Lib ``*/foriceberg/`` folders.

Set ``ICEBERG_GITHUB_ROOT`` to override auto-detected repo root (same as ``FMS2LIB``).
"""
from __future__ import annotations

import os
from pathlib import Path

_SUB = "foriceberg"


def repo_root() -> Path:
    """FMS2Lib repository root (parent of ``script/``, ``src/``, …)."""
    if os.environ.get("ICEBERG_GITHUB_ROOT"):
        return Path(os.environ["ICEBERG_GITHUB_ROOT"]).expanduser().resolve()
    # script/foriceberg/this_file.py -> parents[2]
    return Path(__file__).resolve().parents[2]


def github_root() -> Path:
    """Working directory for ICEBERG drivers (same as ``repo_root()``)."""
    return repo_root()


def script_dir() -> Path:
    return repo_root() / "script" / _SUB


def src_dir() -> Path:
    return repo_root() / "src" / _SUB


def data_dir() -> Path:
    return repo_root() / "data" / _SUB


def mode_dir() -> Path:
    return repo_root() / "mode" / _SUB


def sh_dir() -> Path:
    return repo_root() / "sh" / _SUB


def results_dir() -> Path:
    return repo_root() / "script" / _SUB / "results"


def pubchem_hdf5() -> Path:
    return data_dir() / "pubchem_formulae_inchikey.hdf5"


def data_spec_datasets(*parts: str) -> Path:
    return data_dir() / "spec_datasets" / Path(*parts)


def pythonpath_src() -> str:
    return str(src_dir().resolve())
