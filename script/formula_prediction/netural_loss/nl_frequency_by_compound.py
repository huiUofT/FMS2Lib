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

Count neutral-loss frequencies at the compound level.

For each compound, each Neutral Loss Formula is counted once; the final output
reports how many distinct compounds contain each neutral loss.
"""

import argparse
import os
from collections import Counter
from typing import Optional

import pandas as pd


def load_compound_key(
    expanded_path: str,
    meta_path: Optional[str],
    compound_key: str,
    spectrum_col: str = "Spectrum",
    meta_id_col: str = "FEATURE_ID",
):
    df = pd.read_csv(expanded_path)
    if meta_path and os.path.exists(meta_path):
        meta = pd.read_csv(meta_path)
        if meta_id_col not in meta.columns:
            raise ValueError(f"Metadata CSV is missing column {meta_id_col!r}; available columns: {list(meta.columns)}")
        if compound_key not in meta.columns:
            raise ValueError(f"Metadata CSV is missing compound column {compound_key!r}; available columns: {list(meta.columns)}")
        sub = meta[[meta_id_col, compound_key]].drop_duplicates(subset=[meta_id_col])
        df = df.merge(
            sub,
            left_on=spectrum_col,
            right_on=meta_id_col,
            how="left",
        )
        df = df[df[compound_key].notna() & (df[compound_key].astype(str).str.strip() != "")]
    else:
        if compound_key not in df.columns:
            raise ValueError(
                f"No --meta-csv was provided and expanded CSV has no column {compound_key!r}; "
                f"provide metadata CSV or ensure the expanded CSV contains the compound column. Available columns: {list(df.columns)}"
            )

    return df


def count_nl_per_compound_unique(
    df: pd.DataFrame,
    compound_col: str,
    nl_col: str = "Neutral Loss Formula",
):
    """Deduplicate neutral losses within each compound, then count compounds per neutral loss."""
    valid = df[df[nl_col].notna()].copy()
    valid = valid[valid[nl_col].astype(str).str.strip() != ""]
    valid = valid[valid[nl_col].astype(str) != "No match"]

    per_compound_sets = (
        valid.groupby(compound_col)[nl_col]
        .apply(lambda s: set(s.astype(str).unique()))
    )

    counter: Counter = Counter()
    for _cid, nl_set in per_compound_sets.items():
        for nl in nl_set:
            counter[nl] += 1

    out = (
        pd.Series(counter, name="n_compounds")
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": nl_col})
    )
    return out, len(per_compound_sets)


def main():
    ap = argparse.ArgumentParser(
        description="Count neutral-loss frequencies after deduplicating within each compound."
    )
    ap.add_argument(
        "-i",
        "--expanded-csv",
        required=True,
        help="Expanded CSV written by calculate_inter_fragment_nl.",
    )
    ap.add_argument(
        "-m",
        "--meta-csv",
        default=None,
        help="Optional metadata CSV with FEATURE_ID and a compound identifier column.",
    )
    ap.add_argument(
        "--compound-key",
        default="SMILES",
        help="Compound identifier column, for example SMILES or REF_FORMULA.",
    )
    ap.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output frequency CSV with Neutral Loss Formula and n_compounds columns.",
    )
    args = ap.parse_args()

    df = load_compound_key(
        args.expanded_csv,
        args.meta_csv,
        compound_key=args.compound_key,
    )
    freq_df, n_compounds = count_nl_per_compound_unique(df, args.compound_key)
    freq_df.insert(0, "rank", range(1, len(freq_df) + 1))

    _odir = os.path.dirname(os.path.abspath(args.output))
    if _odir:
        os.makedirs(_odir, exist_ok=True)
    freq_df.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"Compounds with at least one valid neutral loss: {n_compounds}")
    print(f"Distinct neutral-loss formulas: {len(freq_df)}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
