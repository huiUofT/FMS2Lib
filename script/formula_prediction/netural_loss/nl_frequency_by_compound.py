#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计中性丢失在「化合物」维度上的出现频次。

规则：对每个化合物，每种 Neutral Loss Formula 只计一次；
再在全体化合物上统计：有多少个不同的化合物出现过该 NL。
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
            raise ValueError(f"元数据 CSV 缺少列 {meta_id_col!r}，现有: {list(meta.columns)}")
        if compound_key not in meta.columns:
            raise ValueError(f"元数据 CSV 缺少化合物列 {compound_key!r}，现有: {list(meta.columns)}")
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
                f"未提供 --meta-csv 且展开表中无列 {compound_key!r}；"
                f"请传入 FMS2lib.csv 或确保展开表含化合物列。现有列: {list(df.columns)}"
            )

    return df


def count_nl_per_compound_unique(
    df: pd.DataFrame,
    compound_col: str,
    nl_col: str = "Neutral Loss Formula",
):
    """每个化合物内 NL 去重后，统计每个 NL 出现在多少个化合物中。"""
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
        description="按化合物去重后统计中性丢失出现频次（化合物数）"
    )
    ap.add_argument(
        "-i",
        "--expanded-csv",
        required=True,
        help="calculate_inter_fragment_nl 展开后的 CSV",
    )
    ap.add_argument(
        "-m",
        "--meta-csv",
        default=None,
        help="可选：含 FEATURE_ID 与化合物列的元数据（如 FMS2lib.csv）",
    )
    ap.add_argument(
        "--compound-key",
        default="SMILES",
        help="化合物唯一标识列（默认 SMILES；也可用 REF_FORMULA 等）",
    )
    ap.add_argument(
        "-o",
        "--output",
        required=True,
        help="输出频次表 CSV（列: Neutral Loss Formula, n_compounds）",
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

    print(f"化合物数（有至少一条有效 NL 的 group）: {n_compounds}")
    print(f"不同中性丢失类型数: {len(freq_df)}")
    print(f"已写入: {args.output}")


if __name__ == "__main__":
    main()
