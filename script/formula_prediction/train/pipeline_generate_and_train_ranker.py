#!/usr/bin/env python3
"""This code was adapted from the BUDDY implementation developed by
Xing et al.

Original repository:
https://github.com/HuanLab/BUDDY

The BUDDY model was retrained using an in-house MS/MS dataset of
low-fluorinated compounds. Modifications include data preprocessing,
dataset splitting, model training, and performance evaluation.

Modifications by: Tao Young
Date: August 20260819

The original copyright notice and license terms are retained.

Compatibility wrapper — use mgf_to_results.py (same CLI).
"""

from mgf_to_results import main


if __name__ == "__main__":
    raise SystemExit(main())
