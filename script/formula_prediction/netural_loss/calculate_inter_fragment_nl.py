#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Expand inter-fragment neutral losses (one row per loss). Parallelized.

If input lacks a Neutral Loss Formula column, it is filled as
precursor formula minus fragment formula (from-precursor semantics).
"""

import argparse
import os
import re
from typing import Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing as mp
from tqdm import tqdm


def parse_formula(formula_str):
    """Parse formula string to element count dict, e.g. C19H26N4O."""
    if pd.isna(formula_str) or formula_str == 'No match':
        return {}

    pattern = r'([A-Z][a-z]*)(\d*)'
    elements = {}

    for match in re.finditer(pattern, str(formula_str)):
        element = match.group(1)
        count = match.group(2)
        count = int(count) if count else 1
        elements[element] = elements.get(element, 0) + count

    return elements


def formula_to_string(formula_dict):
    """Element dict to Hill-style string; remaining elements sorted alphabetically."""
    if not formula_dict:
        return 'No match'

    standard_order = ['C', 'H', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']

    result_parts = []
    used_elements = set()

    for element in standard_order:
        if element in formula_dict and formula_dict[element] != 0:
            count = formula_dict[element]
            if count == 1:
                result_parts.append(element)
            else:
                result_parts.append(f"{element}{count}")
            used_elements.add(element)

    other_elements = sorted([e for e in formula_dict.keys() if e not in used_elements])
    for element in other_elements:
        if formula_dict[element] != 0:
            count = formula_dict[element]
            if count == 1:
                result_parts.append(element)
            else:
                result_parts.append(f"{element}{count}")

    return ''.join(result_parts) if result_parts else 'No match'


def subtract_formulas(formula1_dict, formula2_dict):
    """formula1 - formula2 in element space."""
    result = formula1_dict.copy()

    for element, count in formula2_dict.items():
        result[element] = result.get(element, 0) - count

    result = {k: v for k, v in result.items() if v > 0}
    return result


def is_subformula(smaller_dict, larger_dict):
    """True if smaller_dict is a subformula of larger_dict."""
    if not smaller_dict:
        return False

    for element, count in smaller_dict.items():
        if larger_dict.get(element, 0) < count:
            return False

    return True


def precompute_precursor_nl(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing/invalid neutral loss as precursor - fragment when possible."""
    df = df.copy()
    if 'Neutral Loss Formula' not in df.columns:
        df['Neutral Loss Formula'] = np.nan

    if 'Precursor Formula' not in df.columns:
        return df

    for idx in df.index:
        cur = df.at[idx, 'Neutral Loss Formula']
        if pd.notna(cur) and str(cur).strip() not in ('', 'No match', 'nan'):
            continue
        prec_s = df.at[idx, 'Precursor Formula']
        frag_s = df.at[idx, 'Formula']
        prec_d = parse_formula(prec_s)
        frag_d = parse_formula(frag_s)
        if not prec_d or not frag_d:
            continue
        nl_d = subtract_formulas(prec_d, frag_d)
        df.at[idx, 'Neutral Loss Formula'] = (
            formula_to_string(nl_d) if nl_d else 'No match'
        )

    return df


def process_spectrum_fragments(group_data):
    """One spectrum: all pairwise fragment losses; returns expanded rows."""
    expanded_rows = []

    fragments = group_data.to_dict('records')

    for frag in fragments:
        if pd.notna(frag.get('Neutral Loss Formula')) and frag.get('Neutral Loss Formula') != 'No match':
            new_row = frag.copy()
            new_row['Neutral Loss Source'] = 'From Precursor'
            expanded_rows.append(new_row)
        else:
            new_row = frag.copy()
            new_row['Neutral Loss Source'] = 'From Precursor'
            expanded_rows.append(new_row)

        frag_formula = frag.get('Formula')
        if pd.isna(frag_formula) or frag_formula == 'No match':
            continue

        frag_dict = parse_formula(frag_formula)
        if not frag_dict:
            continue

        for other_frag in fragments:
            other_formula = other_frag.get('Formula')
            if pd.isna(other_formula) or other_formula == 'No match':
                continue

            if other_frag.get('Peak Index') == frag.get('Peak Index'):
                continue

            other_dict = parse_formula(other_formula)
            if not other_dict:
                continue

            if is_subformula(frag_dict, other_dict):
                nl_dict = subtract_formulas(other_dict, frag_dict)
                if nl_dict:
                    nl_formula = formula_to_string(nl_dict)
                    new_row = frag.copy()
                    new_row['Neutral Loss Formula'] = nl_formula
                    larger_frag = other_formula
                    new_row['Neutral Loss Source'] = f'From Fragment: {larger_frag}'
                    expanded_rows.append(new_row)

            elif is_subformula(other_dict, frag_dict):
                nl_dict = subtract_formulas(frag_dict, other_dict)
                if nl_dict:
                    nl_formula = formula_to_string(nl_dict)
                    new_row = frag.copy()
                    new_row['Neutral Loss Formula'] = nl_formula
                    larger_frag = frag_formula
                    new_row['Neutral Loss Source'] = f'From Fragment: {larger_frag}'
                    expanded_rows.append(new_row)

    return expanded_rows


def run_expand(input_path: str, output_path: str, n_jobs: Optional[int] = None) -> pd.DataFrame:
    n_jobs = n_jobs if n_jobs is not None else min(6, mp.cpu_count() or 1)

    print("=" * 70)
    print("Inter-fragment neutral loss expansion")
    print(f"Parallel jobs: {n_jobs}")
    print("=" * 70)

    print(f"\nReading: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Rows: {len(df):,}")

    df = precompute_precursor_nl(df)

    required_cols = ['Spectrum', 'Formula', 'Peak Index']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    print("\nGrouping by spectrum...")
    grouped = df.groupby('Spectrum')
    n_spectra = len(grouped)
    print(f"Spectra: {n_spectra:,}")

    print("\nRunning...")
    results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(process_spectrum_fragments)(group_data)
        for _, group_data in tqdm(grouped, total=n_spectra, desc="Spectra")
    )

    print("\nMerging...")
    expanded_rows = []
    for result in results:
        expanded_rows.extend(result)

    df_expanded = pd.DataFrame(expanded_rows)

    print(f"\nInput rows:   {len(df):,}")
    print(f"Output rows:  {len(df_expanded):,}")
    print(f"Added:        {len(df_expanded) - len(df):,}")

    _odir = os.path.dirname(os.path.abspath(output_path))
    if _odir:
        os.makedirs(_odir, exist_ok=True)
    df_expanded.to_csv(output_path, index=False)
    print(f"\nWrote: {output_path}")

    if 'Neutral Loss Source' in df_expanded.columns:
        print("\n" + "=" * 70)
        print("Neutral loss source distribution")
        print("=" * 70)
        source_counts = df_expanded['Neutral Loss Source'].value_counts()
        for source, count in source_counts.items():
            print(f"  {source}: {count:,} ({count/len(df_expanded)*100:.2f}%)")

    return df_expanded


def parse_args():
    p = argparse.ArgumentParser(description="Expand inter-fragment neutral losses")
    p.add_argument("-i", "--input", required=True, help="Fragment CSV (subformula output)")
    p.add_argument("-o", "--output", required=True, help="Output expanded CSV")
    p.add_argument("-j", "--jobs", type=int, default=None, help="Parallel jobs (default min(6, CPUs))")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_expand(args.input, args.output, n_jobs=args.jobs)
