#!/usr/bin/env python3
"""Launcher: implementation lives in github/script/ (see README there)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_TARGET = _HERE / "script" / "process_new_dataset_from_mgf_with_parentmass.py"
os.execv(sys.executable, [sys.executable, str(_TARGET)] + sys.argv[1:])
