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

MGF mass calibration: assign_subformula, MS2 preprocessing, hard error filter,
outlier screen (90% CI on raw error), LOWESS (fixed frac=0.1), write calibrated MGF.

  python clean.py --mgf input.mgf --formulas formulas.txt --output out.mgf --msbuddy-dir /path/to/msbuddy-main
  python clean.py --mgf input.mgf --formulas-csv meta.csv --output out.mgf --msbuddy-dir /path/to/msbuddy-main
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

MS2_TOL = 5.0  # default MS2 ppm; override with --ms2-tol
MS1_TOL = 2.0  # default MS1 ppm; override with --ms1-tol (printed; subformula path uses MS2 only)
REL_INT_DENOISE_CUTOFF = 0.01
TOP_N_PER_50_DA = 6
REMOVE_ISOTOPES = True
ISOTOPE_TOL = 0.01
ISOTOPE_RATIO_MAX = 0.5
ERROR_THRESHOLD = 5.0
LOWESS_FRAC = 0.1


def remove_isotope_peaks_from_arrays(mz_array, int_array, tol=0.01, ratio_max=0.5, mass_diff=1.003355):
    """Remove isotope peaks from m/z and intensity arrays."""
    if len(mz_array) == 0:
        return np.array([], dtype=int), np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    sorted_idx = np.argsort(mz_array)
    sorted_mz = mz_array[sorted_idx]
    sorted_int = int_array[sorted_idx]
    keep = np.ones(len(mz_array), dtype=bool)
    for i in range(len(sorted_mz)):
        if not keep[sorted_idx[i]]:
            continue
        for j in range(i + 1, len(sorted_mz)):
            if not keep[sorted_idx[j]]:
                continue
            mz_diff_val = sorted_mz[j] - sorted_mz[i]
            if abs(mz_diff_val - mass_diff) <= tol:
                if sorted_int[j] <= sorted_int[i] * ratio_max:
                    keep[sorted_idx[j]] = False
            elif mz_diff_val > mass_diff + tol:
                break
    return np.arange(len(mz_array))[keep], mz_array[keep], int_array[keep]

try:
    from numba import njit
    @njit
    def ms2_denoise(mz_arr, int_arr, top_n=6, per_mass_range=50):
        """Keep up to top_n peaks per per_mass_range Da bin."""
        group_arr = mz_arr // per_mass_range
        bool_arr = np.array([True] * len(int_arr))
        for group in np.unique(group_arr):
            group_idx = np.where(group_arr == group)[0]
            if len(group_idx) > top_n:
                int_threshold = np.sort(int_arr[group_idx])[-top_n]
                bool_arr[group_idx] = int_arr[group_idx] >= int_threshold
        return bool_arr
except ImportError:
    def ms2_denoise(mz_arr, int_arr, top_n=6, per_mass_range=50):
        """Keep up to top_n peaks per per_mass_range Da bin."""
        group_arr = mz_arr // per_mass_range
        bool_arr = np.array([True] * len(int_arr))
        for group in np.unique(group_arr):
            group_idx = np.where(group_arr == group)[0]
            if len(group_idx) > top_n:
                int_threshold = np.sort(int_arr[group_idx])[-top_n]
                bool_arr[group_idx] = int_arr[group_idx] >= int_threshold
        return bool_arr

def calc_top_n_frag(pre_mz):
    return int(20 + 0.05 * pre_mz)

def preprocess_ms2(mz_array, int_array, precursor_mz, ms2_tol=5.0, ppm=True,
                   rel_int_denoise_cutoff=0.01, top_n_per_50_da=6, remove_isotopes=True,
                   isotope_tol=0.01, isotope_ratio_max=0.5):
    """MS2 preprocess: isotope strip, precursor strip, denoise, top-N per bin, global top-N."""
    if len(mz_array) == 0:
        return np.array([], dtype=int), np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    
    if remove_isotopes:
        idx_array, mz_array, int_array = remove_isotope_peaks_from_arrays(
            mz_array, int_array, tol=isotope_tol, ratio_max=isotope_ratio_max)
    else:
        idx_array = np.arange(len(mz_array))
    
    if len(mz_array) == 0:
        return idx_array, mz_array, int_array
    
    if ppm:
        precursor_tol_da = precursor_mz * ms2_tol / 1e6
    else:
        precursor_tol_da = ms2_tol
    keep_mask = np.abs(mz_array - precursor_mz) > precursor_tol_da
    idx_array = idx_array[keep_mask]
    mz_array = mz_array[keep_mask]
    int_array = int_array[keep_mask]
    
    if len(mz_array) == 0:
        return idx_array, mz_array, int_array
    
    max_int = np.max(int_array)
    if max_int > 0:
        rel_int = int_array / max_int
        keep_mask = rel_int >= rel_int_denoise_cutoff
        idx_array = idx_array[keep_mask]
        mz_array = mz_array[keep_mask]
        int_array = int_array[keep_mask]
    
    if len(mz_array) == 0:
        return idx_array, mz_array, int_array
    
    keep_mask = ms2_denoise(mz_array, int_array, top_n=top_n_per_50_da, per_mass_range=50)
    idx_array = idx_array[keep_mask]
    mz_array = mz_array[keep_mask]
    int_array = int_array[keep_mask]
    
    if len(mz_array) == 0:
        return idx_array, mz_array, int_array
    
    top_n = calc_top_n_frag(precursor_mz)
    if len(int_array) > top_n:
        top_n_idx = np.argsort(int_array)[-top_n:]
        idx_array = idx_array[top_n_idx]
        mz_array = mz_array[top_n_idx]
        int_array = int_array[top_n_idx]
    
    return idx_array, mz_array, int_array

def select_best_match(subform_result, observed_mz):
    """Pick lowest |mass error| candidate."""
    if not subform_result.subform_list:
        return None
    
    best_formula = None
    best_error = float('inf')
    
    for formula_result in subform_result.subform_list:
        abs_error = abs(formula_result.mass_error)
        if abs_error < best_error:
            best_error = abs_error
            
            theoretical_mass = observed_mz + formula_result.mass_error
            
            best_formula = {
                'formula': formula_result.formula,
                'mass': theoretical_mass,
                'mass_error_ppm': formula_result.mass_error_ppm,
                'mass_error_da': formula_result.mass_error
            }
    
    return best_formula

def parse_mgf_file(mgf_path: str) -> List[Dict]:
    """Parse MGF to list of spectrum dicts with peaks."""
    spectra = []
    current_spectrum = {}
    current_peaks = []
    
    with open(mgf_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_stripped = line.strip()
            
            if line_stripped == 'BEGIN IONS':
                current_spectrum = {}
                current_peaks = []
            elif line_stripped == 'END IONS':
                if current_peaks:
                    current_spectrum['peaks'] = current_peaks
                spectra.append(current_spectrum)
                current_spectrum = {}
                current_peaks = []
            elif '=' in line_stripped:
                key, value = line_stripped.split('=', 1)
                current_spectrum[key] = value
            else:
                parts = line_stripped.split()
                if len(parts) >= 2:
                    try:
                        mz = float(parts[0])
                        intensity = float(parts[1])
                        current_peaks.append((mz, intensity))
                    except ValueError:
                        pass
    
    return spectra

def read_formulas(formulas_path: str) -> List[str]:
    """Read one formula per line."""
    formulas = []
    with open(formulas_path, 'r', encoding='utf-8') as f:
        for line in f:
            formula = line.strip()
            if formula:
                formulas.append(formula)
    return formulas


def read_formulas_from_csv(csv_path: str, formula_column: str = "formula") -> List[str]:
    """Formulas from CSV column, same row order as spectra in MGF."""
    df = pd.read_csv(csv_path)
    if formula_column not in df.columns:
        raise ValueError(
            f"Column {formula_column!r} not in CSV; have: {list(df.columns)}"
        )
    return df[formula_column].astype(str).tolist()


def extract_fragments_with_assign_subformula(
    spectra: List[Dict],
    formulas: List[str],
    msbuddy_dir: str,
    ms1_tol: float,
    ms2_tol: float,
) -> pd.DataFrame:
    """assign_subformula with MS2 preprocessing."""
    sys.path.insert(0, msbuddy_dir)
    from msbuddy.cand import assign_subformula

    print(f"  tolerances: ms1={ms1_tol} ppm (informational), ms2={ms2_tol} ppm (preprocess + assign_subformula)")
    
    fragments_data = []
    
    print("\n" + "="*70)
    print("Fragment assignment (MS2 preprocess)")
    print("="*70)
    
    total_peaks_before = 0
    total_peaks_after = 0
    
    for i, (spec, formula) in enumerate(zip(spectra, formulas)):
        if not formula:
            continue
        
        formula = str(formula).strip()
        formula = formula.replace('\n', '').replace('\r', '').replace('\t', '')
        if not formula:
            continue
        
        spectrum_id = spec.get('IDENTIFIER') or spec.get('FEATURE_ID') or spec.get('TITLE', f'spec_{i}')
        precursor_mz = float(spec.get('PRECURSOR_MZ', spec.get('PEPMASS', 0)))
        adduct = spec.get('ADDUCT', '[M+H]+')
        
        adduct = str(adduct).strip()
        if not adduct:
            adduct = '[M+H]+'
        
        peaks = spec.get('peaks', [])
        
        if not peaks:
            continue
        
        raw_mz = np.array([mz for mz, _ in peaks], dtype=np.float32)
        raw_int = np.array([intensity for _, intensity in peaks], dtype=np.float32)
        total_peaks_before += len(raw_mz)
        
        idx_array, processed_mz, processed_int = preprocess_ms2(
            raw_mz, raw_int, precursor_mz,
            ms2_tol=ms2_tol, 
            ppm=True,
            rel_int_denoise_cutoff=REL_INT_DENOISE_CUTOFF,
            top_n_per_50_da=TOP_N_PER_50_DA,
            remove_isotopes=REMOVE_ISOTOPES,
            isotope_tol=ISOTOPE_TOL,
            isotope_ratio_max=ISOTOPE_RATIO_MAX
        )
        
        total_peaks_after += len(processed_mz)
        
        if len(processed_mz) == 0:
            continue
        
        processed_ms2_mz = processed_mz.tolist()
        
        try:
            if not isinstance(formula, str) or len(formula) == 0:
                raise ValueError(f"Invalid formula type or empty: {type(formula)}")
            
            subformula_results = assign_subformula(
                ms2_mz=processed_ms2_mz,
                precursor_formula=formula,
                adduct=adduct,
                ms2_tol=ms2_tol,
                ppm=True,
                dbe_cutoff=0.0
            )
        except Exception as e:
            print(f"  warn: subformula failed for {spectrum_id}: {e}")
            print(f"    formula={repr(formula)}, adduct={repr(adduct)}")
            continue
        
        if not subformula_results:
            continue
        
        for result_idx, subform_result in enumerate(subformula_results):
            if not subform_result.subform_list:
                continue
            
            observed_mz = processed_mz[result_idx]
            best_match = select_best_match(subform_result, observed_mz)
            
            if best_match is None:
                continue
            
            original_peak_idx = idx_array[result_idx] if result_idx < len(idx_array) else result_idx
            theoretical_mass = best_match['mass']
            mass_error_ppm = best_match['mass_error_ppm']
            
            fragments_data.append({
                'Spectrum': spectrum_id,
                'Spectrum_Index': i,
                'Precursor_mz': precursor_mz,
                'm/z': observed_mz,
                'm/z_original_idx': original_peak_idx,
                'Formula': best_match['formula'],
                'Theoretical_mass': theoretical_mass,
                'Mass_Error_ppm': mass_error_ppm,
                'Intensity': processed_int[result_idx] if result_idx < len(processed_int) else 0.0
            })
    
    print(f"\nPreprocess stats:")
    print(f"  peaks before: {total_peaks_before:,}")
    print(f"  peaks after:  {total_peaks_after:,}")
    print(f"  removed: {total_peaks_before - total_peaks_after:,} ({(total_peaks_before - total_peaks_after)/total_peaks_before*100:.2f}%)")
    print(f"  fragment rows: {len(fragments_data):,}")
    
    return pd.DataFrame(fragments_data)

def dynamic_outlier_removal(df: pd.DataFrame, frac: float = LOWESS_FRAC) -> pd.DataFrame:
    """Remove points outside local 90% CI on raw Mass_Error_ppm."""
    print("\n" + "="*70)
    print("Outlier screen (90% CI on raw error)")
    print("="*70)
    
    df_sorted = df.sort_values('m/z').copy()
    mz_all = df_sorted['m/z'].values
    error_all = df_sorted['Mass_Error_ppm'].values
    n_total = len(df_sorted)
    
    print(f"points: {n_total:,}")
    print(f"LOWESS frac = {frac}")
    
    window_size = int(n_total * frac)
    print(f"local window ~ {window_size} points")
    
    print("\n90% CI per point (raw error)...")
    keep_mask = np.ones(n_total, dtype=bool)
    ci_lower = np.zeros(n_total)
    ci_upper = np.zeros(n_total)
    
    for i in range(n_total):
        half_window = window_size // 2
        start_idx = max(0, i - half_window)
        end_idx = min(n_total, i + half_window)
        
        neighborhood_errors = error_all[start_idx:end_idx]
        
        if len(neighborhood_errors) > 10:
            q5 = np.percentile(neighborhood_errors, 5)
            q95 = np.percentile(neighborhood_errors, 95)
            
            ci_lower[i] = q5
            ci_upper[i] = q95
            
            if error_all[i] < q5 or error_all[i] > q95:
                keep_mask[i] = False
        else:
            ci_lower[i] = error_all[i]
            ci_upper[i] = error_all[i]
            keep_mask[i] = True
        
        if (i + 1) % 10000 == 0:
            print(f"  ... {i+1:,}/{n_total:,}")
    
    df_clean = df_sorted[keep_mask].copy()
    
    n_kept = keep_mask.sum()
    n_removed = (~keep_mask).sum()
    print(f"\nOutlier screen:")
    print(f"  before: {n_total:,}")
    print(f"  kept:   {n_kept:,} ({n_kept/n_total*100:.2f}%)")
    print(f"  drop:   {n_removed:,} ({n_removed/n_total*100:.2f}%)")
    
    return df_clean

def lowess_calibration(df: pd.DataFrame, frac: float = LOWESS_FRAC) -> Tuple[np.ndarray, np.ndarray]:
    """LOWESS smooth Mass_Error_ppm vs m/z; fixed frac."""
    from statsmodels.nonparametric.smoothers_lowess import lowess
    
    print("\n" + "="*70)
    print(f"LOWESS calibration (frac={frac})")
    print("="*70)
    
    df_sorted = df.sort_values('m/z').copy()
    mz_all = df_sorted['m/z'].values
    error_all = df_sorted['Mass_Error_ppm'].values
    
    if len(df_sorted) > 50000:
        print(f"Large table ({len(df_sorted):,} rows); subsample 50000 for LOWESS fit...")
        sample_indices = np.linspace(0, len(df_sorted) - 1, 50000, dtype=int)
        mz_fit = mz_all[sample_indices]
        error_fit = error_all[sample_indices]
    else:
        mz_fit = mz_all
        error_fit = error_all
    
    print(f"LOWESS fit (frac={frac})...")
    lowess_result = lowess(
        error_fit,
        mz_fit,
        frac=frac,
        it=3,
        return_sorted=True
    )
    
    lowess_mz = lowess_result[:, 0]
    lowess_error = lowess_result[:, 1]
    error_fitted_all = np.interp(mz_all, lowess_mz, lowess_error)
    
    ss_res = np.sum((error_all - error_fitted_all) ** 2)
    ss_tot = np.sum((error_all - np.mean(error_all)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    rmse = np.sqrt(np.mean((error_all - error_fitted_all) ** 2))
    
    print(f"LOWESS result:")
    print(f"  points: {len(df_sorted):,}")
    print(f"  R² = {r2:.4f}")
    print(f"  RMSE = {rmse:.4f} ppm")
    
    return lowess_error, lowess_mz

def apply_calibration_to_fragments(df: pd.DataFrame, lowess_fit: np.ndarray, mz_values: np.ndarray) -> pd.DataFrame:
    """Apply LOWESS delta to m/z; recompute ppm error from theory when available."""
    from scipy.interpolate import interp1d
    
    interp_func = interp1d(mz_values, lowess_fit, kind='linear', 
                           bounds_error=False, fill_value='extrapolate')
    
    df_calibrated = df.copy()
    df_calibrated['LOWESS_delta_ppm'] = interp_func(df_calibrated['m/z'].values)
    df_calibrated['m/z_corrected'] = df_calibrated['m/z'] * (1 + df_calibrated['LOWESS_delta_ppm'] / 1e6)
    
    if 'Theoretical_mass' in df_calibrated.columns:
        df_calibrated['Mass_Error_ppm_corrected'] = (
            (df_calibrated['Theoretical_mass'] - df_calibrated['m/z_corrected']) / 
            df_calibrated['m/z_corrected'] * 1e6
        )
    else:
        df_calibrated['Mass_Error_ppm_corrected'] = df_calibrated['Mass_Error_ppm'] - df_calibrated['LOWESS_delta_ppm']
    
    return df_calibrated

def filter_fragments_by_error(df: pd.DataFrame, threshold: float = ERROR_THRESHOLD) -> pd.DataFrame:
    """Hard filter on |Mass_Error_ppm| before outlier screen."""
    print("\n" + "="*70)
    print(f"Hard filter: drop |Mass Error| > {threshold} ppm")
    print("="*70)
    
    before_count = len(df)
    mask_keep = df['Mass_Error_ppm'].abs() <= threshold
    df_filtered = df[mask_keep].copy()
    after_count = len(df_filtered)
    
    print(f"  before: {before_count:,}")
    print(f"  after:  {after_count:,}")
    print(f"  drop:   {before_count - after_count:,} ({(before_count - after_count)/before_count*100:.2f}%)")
    
    return df_filtered

def update_mgf_with_corrected_masses(mgf_input: str, mgf_output: str, df_calibrated: pd.DataFrame):
    """Rewrite MGF peaks using calibrated m/z; drops unmatched peaks."""
    print("\n" + "="*70)
    print("Write calibrated MGF")
    print("="*70)
    
    spectrum_to_corrected = {}
    for _, row in df_calibrated.iterrows():
        spectrum_idx = int(row.get('Spectrum_Index', -1))
        if spectrum_idx < 0:
            spectrum_id = str(row['Spectrum'])
            if spectrum_id not in spectrum_to_corrected:
                spectrum_to_corrected[spectrum_id] = {}
            original_mz = float(row['m/z'])
            corrected_mz = float(row['m/z_corrected'])
            spectrum_to_corrected[spectrum_id][original_mz] = corrected_mz
        else:
            if spectrum_idx not in spectrum_to_corrected:
                spectrum_to_corrected[spectrum_idx] = {}
            original_mz = float(row['m/z'])
            corrected_mz = float(row['m/z_corrected'])
            spectrum_to_corrected[spectrum_idx][original_mz] = corrected_mz
    
    with open(mgf_input, 'r', encoding='utf-8') as f_in, \
         open(mgf_output, 'w', encoding='utf-8') as f_out:
        
        in_spectrum = False
        current_spectrum_idx = -1
        current_spectrum_id = None
        current_peaks = []
        spectrum_count = 0
        
        for line in f_in:
            line_stripped = line.strip()
            
            if line_stripped == 'BEGIN IONS':
                in_spectrum = True
                current_spectrum_idx = spectrum_count
                current_spectrum_id = None
                current_peaks = []
                spectrum_count += 1
                f_out.write(line)
            elif line_stripped == 'END IONS':
                mz_map = None
                if current_spectrum_idx in spectrum_to_corrected:
                    mz_map = spectrum_to_corrected[current_spectrum_idx]
                elif current_spectrum_id and current_spectrum_id in spectrum_to_corrected:
                    mz_map = spectrum_to_corrected[current_spectrum_id]
                
                if mz_map:
                    for peak_line in current_peaks:
                        parts = peak_line.strip().split()
                        if len(parts) >= 2:
                            try:
                                original_mz = float(parts[0])
                                intensity = parts[1]
                                
                                matched_corrected_mz = None
                                min_diff = float('inf')
                                for orig_mz, corr_mz in mz_map.items():
                                    diff = abs(original_mz - orig_mz)
                                    if diff < min_diff and diff < 0.0001:
                                        min_diff = diff
                                        matched_corrected_mz = corr_mz
                                
                                if matched_corrected_mz is not None:
                                    f_out.write(f"{matched_corrected_mz:.6f} {intensity}\n")
                            except (ValueError, IndexError):
                                f_out.write(peak_line)
                        else:
                            f_out.write(peak_line)
                else:
                    for peak_line in current_peaks:
                        f_out.write(peak_line)
                
                f_out.write("END IONS\n\n")
                in_spectrum = False
                current_spectrum_idx = -1
                current_spectrum_id = None
                current_peaks = []
            elif in_spectrum:
                if '=' in line_stripped:
                    key, value = line_stripped.split('=', 1)
                    if key in ['IDENTIFIER', 'FEATURE_ID', 'TITLE']:
                        current_spectrum_id = value
                    f_out.write(line)
                else:
                    current_peaks.append(line)
            else:
                f_out.write(line)
    
    print(f"  output: {mgf_output}")
    print(f"  spectra: {spectrum_count}")

def parse_arguments():
    """CLI."""
    parser = argparse.ArgumentParser(
        description='MGF mass calibration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
  python clean.py --mgf input.mgf --formulas formulas.txt --output out.mgf --msbuddy-dir /path/to/msbuddy-main
  python clean.py --mgf input.mgf --formulas-csv metadata.csv --output out.mgf --msbuddy-dir /path/to/msbuddy-main
        """
    )
    
    parser.add_argument('--mgf', type=str, required=True,
                        help='Input MGF path')
    formula_group = parser.add_mutually_exclusive_group(required=True)
    formula_group.add_argument(
        '--formulas',
        type=str,
        default=None,
        help='Text file, one formula per line (same order as MGF)',
    )
    formula_group.add_argument(
        '--formulas-csv',
        type=str,
        default=None,
        dest='formulas_csv',
        help='CSV with a formula column (default column name: formula)',
    )
    parser.add_argument(
        '--formula-column',
        type=str,
        default='formula',
        help='Formula column for --formulas-csv (default: formula)',
    )
    parser.add_argument('--output', type=str, required=True,
                        help='Output MGF path')
    parser.add_argument('--msbuddy-dir', type=str, required=True,
                        help='msbuddy package root (contains msbuddy/)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='CSV output dir (fragments_*.csv); default next to --output MGF')
    parser.add_argument('--error-threshold', type=float, default=ERROR_THRESHOLD,
                        help=f'|ppm| hard filter (default {ERROR_THRESHOLD})')
    parser.add_argument('--lowess-frac', type=float, default=LOWESS_FRAC,
                        help=f'LOWESS frac (default {LOWESS_FRAC})')
    parser.add_argument(
        '--ms1-tol',
        type=float,
        default=MS1_TOL,
        help=f'MS1 tolerance in ppm (default {MS1_TOL}; informational for this pipeline)',
    )
    parser.add_argument(
        '--ms2-tol',
        type=float,
        default=MS2_TOL,
        help=f'MS2 tolerance in ppm for preprocess + assign_subformula (default {MS2_TOL})',
    )

    return parser.parse_args()

def main():
    args = parse_arguments()
    
    print("="*70)
    print("MGF mass calibration")
    print("="*70)
    print(f"Tolerances: ms1_tol={args.ms1_tol} ppm, ms2_tol={args.ms2_tol} ppm")

    if not os.path.exists(args.mgf):
        print(f"Error: MGF not found: {args.mgf}")
        sys.exit(1)
    
    if args.formulas_csv:
        if not os.path.exists(args.formulas_csv):
            print(f"Error: formulas CSV not found: {args.formulas_csv}")
            sys.exit(1)
    else:
        if not os.path.exists(args.formulas):
            print(f"Error: formulas file not found: {args.formulas}")
            sys.exit(1)
    
    if not os.path.exists(args.msbuddy_dir):
        print(f"Error: msbuddy dir not found: {args.msbuddy_dir}")
        sys.exit(1)
    
    if args.output_dir is None:
        output_dir = os.path.dirname(args.output)
        if not output_dir:
            output_dir = '.'
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    global ERROR_THRESHOLD, LOWESS_FRAC
    ERROR_THRESHOLD = args.error_threshold
    LOWESS_FRAC = args.lowess_frac
    
    print("\n1. Load MGF and formulas...")
    print(f"   MGF: {args.mgf}")
    if args.formulas_csv:
        print(f"   CSV: {args.formulas_csv} (column {args.formula_column})")
        try:
            formulas = read_formulas_from_csv(args.formulas_csv, args.formula_column)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        print(f"   formulas: {args.formulas}")
        formulas = read_formulas(args.formulas)
    spectra = parse_mgf_file(args.mgf)
    print(f"   spectra: {len(spectra)}")
    print(f"   formulas: {len(formulas)}")
    
    if len(spectra) != len(formulas):
        print(f"   warn: spectrum count {len(spectra)} != formula count {len(formulas)}")
    
    df_fragments = extract_fragments_with_assign_subformula(
        spectra,
        formulas,
        args.msbuddy_dir,
        ms1_tol=args.ms1_tol,
        ms2_tol=args.ms2_tol,
    )
    
    if len(df_fragments) == 0:
        print("Error: no fragments extracted")
        sys.exit(1)
    
    fragments_original_path = os.path.join(output_dir, 'fragments_original.csv')
    df_fragments.to_csv(fragments_original_path, index=False)
    print(f"\n   wrote {fragments_original_path}")
    
    df_filtered = filter_fragments_by_error(df_fragments, threshold=ERROR_THRESHOLD)
    
    df_clean = dynamic_outlier_removal(df_filtered, frac=LOWESS_FRAC)
    
    lowess_fit, mz_values = lowess_calibration(df_clean, frac=LOWESS_FRAC)
    df_calibrated = apply_calibration_to_fragments(df_clean, lowess_fit, mz_values)
    
    df_final = df_calibrated.copy()
    
    fragments_calibrated_path = os.path.join(output_dir, 'fragments_calibrated.csv')
    df_final.to_csv(fragments_calibrated_path, index=False)
    print(f"\n   wrote {fragments_calibrated_path}")
    
    update_mgf_with_corrected_masses(args.mgf, args.output, df_final)
    
    print("\n" + "="*70)
    print("Done")
    print("="*70)
    print("Outputs:")
    print(f"  MGF:         {args.output}")
    print(f"  fragments:   {fragments_original_path}")
    print(f"  calibrated:  {fragments_calibrated_path}")

if __name__ == '__main__':
    main()
