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
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = str(Path(__file__).resolve().parents[3])
MSBUDDY_DIR = os.path.join(BASE_DIR, "src")
MGF_FILE = None  # Supplied through the required --mgf argument.
OUTPUT_DIR = os.path.join(BASE_DIR, "scratch", "subformula")
TEMP_DIR = os.path.join(OUTPUT_DIR, "temp")

NUM_CORES = 6
SAVE_INTERVAL = 30000

# On Windows, worker processes re-import this module; set MSBUDDY_DIR via CLI then
# add msbuddy-main to sys.path before importing msbuddy.
assign_subformula = None
calc_formula_dbe = None
read_formula = None

try:
    from numba import njit
except Exception:
    def njit(*args, **kwargs):
        def deco(fn):
            return fn
        return deco

def remove_isotope_peaks_from_arrays(mz_array, int_array, tol=0.01, ratio_max=0.5, mass_diff=1.003355):
    """Remove isotope peaks; keep float64 m/z and intensity."""
    if len(mz_array) == 0:
        return np.array([], dtype=int), np.array([], dtype=np.float64), np.array([], dtype=np.float64)
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

def calc_top_n_frag(pre_mz):
    return int(20 + 0.05 * pre_mz)

def preprocess_ms2(mz_array, int_array, precursor_mz, ms2_tol=5.0, ppm=True,
                   rel_int_denoise_cutoff=0.01, top_n_per_50_da=6, remove_isotopes=True,
                   isotope_tol=0.01, isotope_ratio_max=0.5):
    """MS2 preprocess: isotopes, precursor strip, denoise, top-N per bin, global top-N."""
    if len(mz_array) == 0:
        return np.array([], dtype=int), np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    
    if remove_isotopes:
        idx_array, mz_array, int_array = remove_isotope_peaks_from_arrays(
            mz_array, int_array, tol=isotope_tol, ratio_max=isotope_ratio_max)
    else:
        idx_array = np.arange(len(mz_array))
    
    if len(mz_array) == 0:
        return idx_array, mz_array, int_array
    
    mz_tol = precursor_mz * ms2_tol * 1e-6 if ppm else ms2_tol
    mask = mz_array < (precursor_mz - 1.5 - mz_tol)
    idx_array, mz_array, int_array = idx_array[mask], mz_array[mask], int_array[mask]
    
    if len(mz_array) == 0:
        return idx_array, mz_array, int_array
    
    sorted_int = np.sort(int_array)
    mask = int_array >= rel_int_denoise_cutoff * sorted_int[-1]
    idx_array, mz_array, int_array = idx_array[mask], mz_array[mask], int_array[mask]
    
    if len(mz_array) == 0:
        return idx_array, mz_array, int_array
    
    if top_n_per_50_da > 1:
        mask = ms2_denoise(mz_array, int_array, top_n=top_n_per_50_da)
        idx_array, mz_array, int_array = idx_array[mask], mz_array[mask], int_array[mask]
    
    if len(mz_array) == 0:
        return idx_array, mz_array, int_array
    
    top_n_frag = calc_top_n_frag(precursor_mz)
    if len(mz_array) > top_n_frag:
        sorted_idx = np.argsort(int_array)
        mask = int_array >= int_array[sorted_idx[-top_n_frag]]
        idx_array, mz_array, int_array = idx_array[mask], mz_array[mask], int_array[mask]
    
    return idx_array, mz_array, int_array

def read_mgf_manual(mgf_file: str) -> List[Dict]:
    """Parse MGF into list of dicts with params and numpy m/z, intensity arrays."""
    spectra = []
    
    with open(mgf_file, 'r', encoding='utf-8', errors='ignore') as f:
        inside_ions = False
        params = {}
        mz_list = []
        int_list = []
        
        for line in f:
            line = line.strip()
            
            if line == "BEGIN IONS":
                if inside_ions and len(mz_list) > 0:
                    spectrum = {
                        'params': params.copy(),
                        'm/z array': np.array(mz_list, dtype=np.float64),
                        'intensity array': np.array(int_list, dtype=np.float64)
                    }
                    spectra.append(spectrum)
                
                inside_ions = True
                params = {}
                mz_list = []
                int_list = []
            
            elif line == "END IONS":
                if inside_ions:
                    if len(mz_list) > 0:
                        spectrum = {
                            'params': params.copy(),
                            'm/z array': np.array(mz_list, dtype=np.float64),
                            'intensity array': np.array(int_list, dtype=np.float64)
                        }
                        spectra.append(spectrum)
                    inside_ions = False
                    params = {}
                    mz_list = []
                    int_list = []
            
            elif inside_ions:
                if '=' in line:
                    parts = line.split('=', 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        params[key] = value
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            mz_val = float(parts[0])
                            int_val = float(parts[1])
                            if mz_val > 0:
                                mz_list.append(mz_val)
                                int_list.append(int_val)
                        except (ValueError, IndexError):
                            pass
    
    return spectra

def add_radical_label_to_results(subformula_results, ms2_mz, adduct):
    """Add DBE and radical flag; keep best (min |error|) candidate per peak."""
    from msbuddy.cand import check_adduct
    
    enhanced_results = []
    for result in subformula_results:
        peak_idx = result.idx
        mz = ms2_mz[peak_idx]
        
        if len(result.subform_list) == 0:
            enhanced_results.append({
                'peak_idx': peak_idx, 'm/z': mz, 'formula': None, 'num_candidates': 0,
                'dbe': None, 'is_radical': None, 'mass_error_ppm': None, 'mass_error_da': None})
            continue
        
        best_formula = None
        best_error = float('inf')
        num_candidates = len(result.subform_list)
        
        for formula_result in result.subform_list:
            formula_array = read_formula(formula_result.formula)
            if formula_array is None:
                continue
            abs_error = abs(formula_result.mass_error)
            if abs_error < best_error:
                best_error = abs_error
                best_formula = {
                    'formula': formula_result.formula,
                    'formula_array': formula_array,
                    'mass_error_ppm': formula_result.mass_error_ppm,
                    'mass_error_da': formula_result.mass_error
                }
        
        if best_formula is None:
            enhanced_results.append({
                'peak_idx': peak_idx, 'm/z': mz, 'formula': None, 'num_candidates': num_candidates,
                'dbe': None, 'is_radical': None, 'mass_error_ppm': None, 'mass_error_da': None})
        else:
            dbe = calc_formula_dbe(best_formula['formula_array'])
            enhanced_results.append({
                'peak_idx': peak_idx, 'm/z': mz, 'formula': best_formula['formula'],
                'num_candidates': num_candidates, 'dbe': float(dbe), 'is_radical': (dbe % 1 == 0),
                'mass_error_ppm': best_formula['mass_error_ppm'],
                'mass_error_da': best_formula['mass_error_da']})
    
    return enhanced_results

def _ensure_msbuddy_for_worker():
    """Worker entry: import msbuddy after adding msbuddy-main to sys.path (spawn)."""
    global assign_subformula, calc_formula_dbe, read_formula
    if assign_subformula is not None:
        return
    _here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(_here, "..", "..", "msbuddy-main"),
        os.path.join(_here, "..", "msbuddy-main"),
        os.path.join(_here, "msbuddy-main"),
    ]
    for c in candidates:
        c = os.path.abspath(c)
        if os.path.isdir(os.path.join(c, "msbuddy")):
            if c not in sys.path:
                sys.path.insert(0, c)
            break
    from msbuddy.cand import assign_subformula as _asf
    from msbuddy.base import calc_formula_dbe as _dbe
    from msbuddy.utils import read_formula as _rf

    assign_subformula = _asf
    calc_formula_dbe = _dbe
    read_formula = _rf


def process_single_spectrum(args):
    """One spectrum; used by multiprocessing.Pool."""
    spec_idx, spectrum, formulas_dict, config = args

    try:
        _ensure_msbuddy_for_worker()
        params = spectrum.get('params', {})
        identifier = params.get('IDENTIFIER', params.get('FEATURE_ID', f'Spectrum_{spec_idx}'))
        precursor_mz = float(params.get('PRECURSOR_MZ', params.get('PEPMASS', 0)))
        adduct = params.get('ADDUCT', '[M+H]+')
        
        if identifier in formulas_dict:
            precursor_formula = formulas_dict[identifier]
        elif spec_idx in formulas_dict:
            precursor_formula = formulas_dict[spec_idx]
        else:
            return None
        
        if not precursor_formula or precursor_formula == '' or precursor_formula.strip() == '':
            return None
        
        raw_mz = spectrum['m/z array'].copy() if isinstance(spectrum['m/z array'], np.ndarray) else np.array(spectrum['m/z array'], dtype=np.float64)
        raw_int = spectrum['intensity array'].copy() if isinstance(spectrum['intensity array'], np.ndarray) else np.array(spectrum['intensity array'], dtype=np.float64)
        
        if config.get('NO_PREPROCESS', False):
            mz_tol = precursor_mz * config['MS2_TOL'] * 1e-6 if config['PPM'] else config['MS2_TOL']
            mask = raw_mz < (precursor_mz - 1.5 - mz_tol)
            idx_array = np.arange(len(raw_mz), dtype=int)[mask]
            processed_mz = raw_mz[mask]
            processed_int = raw_int[mask]
        else:
            idx_array, processed_mz, processed_int = preprocess_ms2(
            raw_mz, raw_int, precursor_mz, config['MS2_TOL'], config['PPM'],
            config['REL_INT_DENOISE_CUTOFF'], config['TOP_N_PER_50_DA'],
            config['REMOVE_ISOTOPES'], config['ISOTOPE_TOL'], config['ISOTOPE_RATIO_MAX'])
        
        if len(processed_mz) == 0:
            return None
        
        processed_ms2_mz = processed_mz.tolist()
        
        subformula_results = assign_subformula(
            ms2_mz=processed_ms2_mz,
            precursor_formula=precursor_formula,
            adduct=adduct,
            ms2_tol=config['MS2_TOL'],
            ppm=config['PPM'],
            dbe_cutoff=config['DBE_CUTOFF']
        )
        
        if subformula_results is None:
            return None
        
        results = add_radical_label_to_results(subformula_results, processed_ms2_mz, adduct)
        
        matched_peaks = sum(1 for r in results if r['formula'] is not None)
        radical_count = sum(1 for r in results if r['formula'] is not None and r['is_radical'])
        non_radical_count = matched_peaks - radical_count
        
        fragments = []
        for r in results:
            mass_error_ppm_val = r['mass_error_ppm'] if r['mass_error_ppm'] is not None else None
            mass_error_da_val = r['mass_error_da'] if r['mass_error_da'] is not None else None
            
            fragments.append({
                'Spectrum': identifier,
                'Identifier': identifier,
                'Precursor Formula': precursor_formula,
                'Precursor m/z': precursor_mz,
                'Adduct': adduct,
                'Peak Index': r['peak_idx'],
                'm/z': r['m/z'],
                'Formula': r['formula'] if r['formula'] else 'No match',
                'Num Candidates': r['num_candidates'],
                'DBE': r['dbe'] if r['dbe'] is not None else '',
                'Is Radical': 'Yes' if r['is_radical'] else 'No' if r['is_radical'] is not None else '',
                'Mass_Error_ppm': mass_error_ppm_val,
                'Mass_Error_mDa': mass_error_da_val * 1000 if mass_error_da_val is not None else None,
                'Mass Error (ppm)': f"{mass_error_ppm_val:.3f}" if mass_error_ppm_val is not None else '',
                'Mass Error (mDa)': f"{mass_error_da_val*1000:.3f}" if mass_error_da_val is not None else ''
            })
        
        return {
            'spectrum_idx': spec_idx + 1,
            'identifier': identifier,
            'precursor_mz': precursor_mz,
            'precursor_formula': precursor_formula,
            'adduct': adduct,
            'raw_peaks': len(raw_mz),
            'processed_peaks': len(processed_mz),
            'matched_peaks': matched_peaks,
            'radical_count': radical_count,
            'non_radical_count': non_radical_count,
            'fragments': fragments
        }
    except Exception as e:
        return None

LOWESS_FRAC = 0.1
ERROR_THRESHOLD = 5.0

def dynamic_outlier_removal(df: pd.DataFrame, frac: float = LOWESS_FRAC) -> pd.DataFrame:
    """Remove points outside local 90% CI on residual from LOWESS(Mass_Error_ppm vs m/z)."""
    from statsmodels.nonparametric.smoothers_lowess import lowess
    
    print("\n" + "="*70)
    print("Outlier screen (90% CI on residual)")
    print("="*70)
    
    df_valid = df[df['Mass_Error_ppm'].notna()].copy()
    if len(df_valid) == 0:
        print("  warn: no Mass_Error_ppm; skip outlier step")
        return df
    
    df_sorted = df_valid.sort_values('m/z').copy()
    
    lowess_result = lowess(
        df_sorted['Mass_Error_ppm'].values,
        df_sorted['m/z'].values,
        frac=frac,
        return_sorted=False
    )
    df_sorted['LOWESS_fit'] = lowess_result
    
    df_sorted['Residual'] = df_sorted['Mass_Error_ppm'] - df_sorted['LOWESS_fit']
    
    window_size = int(len(df_sorted) * frac)
    if window_size < 10:
        window_size = 10
    keep_mask = np.ones(len(df_sorted), dtype=bool)
    
    for i in range(len(df_sorted)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(df_sorted), i + window_size // 2)
        window_residuals = df_sorted['Residual'].iloc[start_idx:end_idx].values
        
        if len(window_residuals) > 10:
            q5 = np.percentile(window_residuals, 5)
            q95 = np.percentile(window_residuals, 95)
            ci_lower = df_sorted['LOWESS_fit'].iloc[i] + q5
            ci_upper = df_sorted['LOWESS_fit'].iloc[i] + q95
            
            mass_error = df_sorted['Mass_Error_ppm'].iloc[i]
            if mass_error < ci_lower or mass_error > ci_upper:
                keep_mask[i] = False
    
    df_clean = df_sorted[keep_mask].copy()
    
    df_no_error = df[df['Mass_Error_ppm'].isna()].copy()
    df_final = pd.concat([df_clean, df_no_error], ignore_index=True)
    
    print(f"  before: {len(df_valid):,}")
    print(f"  kept:   {len(df_clean):,}")
    print(f"  drop:   {len(df_valid) - len(df_clean):,} ({(len(df_valid) - len(df_clean))/len(df_valid)*100:.2f}%)")
    
    return df_final

def lowess_calibration(df: pd.DataFrame, frac: float = LOWESS_FRAC):
    """LOWESS smooth Mass_Error_ppm vs m/z."""
    from statsmodels.nonparametric.smoothers_lowess import lowess
    
    print("\n" + "="*70)
    print("LOWESS calibration")
    print("="*70)
    
    df_valid = df[df['Mass_Error_ppm'].notna()].copy()
    if len(df_valid) == 0:
        print("  warn: no Mass_Error_ppm; skip calibration")
        return None, None, df
    
    df_sorted = df_valid.sort_values('m/z').copy()
    
    lowess_result = lowess(
        df_sorted['Mass_Error_ppm'].values,
        df_sorted['m/z'].values,
        frac=frac,
        return_sorted=False
    )
    
    print(f"  LOWESS frac = {frac}")
    print(f"  points: {len(df_sorted):,}")
    
    return lowess_result, df_sorted['m/z'].values, df_sorted

def apply_calibration_to_fragments(df: pd.DataFrame, lowess_fit: np.ndarray, mz_values: np.ndarray) -> pd.DataFrame:
    """Interpolate LOWESS fit; update m/z and residual ppm (same convention as subformula/clean)."""
    from scipy.interpolate import interp1d
    
    df_valid = df[df['Mass_Error_ppm'].notna()].copy()
    df_no_error = df[df['Mass_Error_ppm'].isna()].copy()
    
    if len(df_valid) == 0:
        df_result = df.copy()
        df_result['LOWESS_delta_ppm'] = None
        df_result['m/z_corrected'] = df_result['m/z']
        df_result['Mass_Error_ppm_corrected'] = None
        return df_result
    
    interp_func = interp1d(mz_values, lowess_fit, kind='linear', 
                           bounds_error=False, fill_value='extrapolate')
    
    df_calibrated = df_valid.copy()
    df_calibrated['LOWESS_delta_ppm'] = interp_func(df_calibrated['m/z'].values)
    df_calibrated['m/z_corrected'] = df_calibrated['m/z'] * (1 - df_calibrated['LOWESS_delta_ppm'] / 1e6)
    df_calibrated['Mass_Error_ppm_corrected'] = df_calibrated['Mass_Error_ppm'] - df_calibrated['LOWESS_delta_ppm']
    
    if len(df_no_error) > 0:
        df_no_error = df_no_error.copy()
        df_no_error['LOWESS_delta_ppm'] = np.nan
        df_no_error['m/z_corrected'] = df_no_error['m/z']
        df_no_error['Mass_Error_ppm_corrected'] = np.nan
        df_calibrated = pd.concat([df_calibrated, df_no_error], ignore_index=True)
    
    return df_calibrated

def filter_fragments_by_error(df: pd.DataFrame, threshold: float = ERROR_THRESHOLD) -> pd.DataFrame:
    """Drop rows with |Mass_Error_ppm_corrected| > threshold."""
    print("\n" + "="*70)
    print(f"Filter |corrected error| > {threshold} ppm")
    print("="*70)
    
    df_with_correction = df[df['Mass_Error_ppm_corrected'].notna()].copy()
    df_no_correction = df[df['Mass_Error_ppm_corrected'].isna()].copy()
    
    if len(df_with_correction) == 0:
        print("  warn: no corrected ppm column")
        return df
    
    before_count = len(df_with_correction)
    mask_keep = df_with_correction['Mass_Error_ppm_corrected'].abs() <= threshold
    df_filtered = df_with_correction[mask_keep].copy()
    after_count = len(df_filtered)
    
    print(f"  before: {before_count:,}")
    print(f"  after:  {after_count:,}")
    print(f"  drop:   {before_count - after_count:,} ({(before_count - after_count)/before_count*100:.2f}%)")
    
    if len(df_no_correction) > 0:
        df_filtered = pd.concat([df_filtered, df_no_correction], ignore_index=True)
    
    return df_filtered

def save_temp_files(part_num, summary_data, fragments_data, temp_dir):
    """Write one chunk to temp CSVs."""
    df_summary = pd.DataFrame(summary_data)
    df_fragments = pd.DataFrame(fragments_data)
    
    summary_file = os.path.join(temp_dir, f"summary_part_{part_num:05d}.csv")
    fragments_file = os.path.join(temp_dir, f"fragments_part_{part_num:05d}.csv")
    
    df_summary.to_csv(summary_file, index=False, encoding='utf-8-sig')
    df_fragments.to_csv(fragments_file, index=False, encoding='utf-8-sig')
    
    return summary_file, fragments_file

def merge_temp_files(temp_dir, output_dir, mgf_file):
    """Concat temp part files into final summary and fragments CSV."""
    import glob
    
    mgf_basename = os.path.splitext(os.path.basename(mgf_file))[0]
    
    summary_files = sorted(glob.glob(os.path.join(temp_dir, "summary_part_*.csv")))
    if len(summary_files) > 0:
        print(f"\nMerging {len(summary_files)} summary parts...")
        df_summary_list = [pd.read_csv(f) for f in tqdm(summary_files, desc="summary parts")]
        df_summary_merged = pd.concat(df_summary_list, ignore_index=True)
        
        output_summary = os.path.join(output_dir, f"{mgf_basename}_summary.csv")
        df_summary_merged.to_csv(output_summary, index=False, encoding='utf-8-sig')
        print(f"  wrote summary: {output_summary} ({len(df_summary_merged)} rows)")
    
    fragments_files = sorted(glob.glob(os.path.join(temp_dir, "fragments_part_*.csv")))
    if len(fragments_files) > 0:
        print(f"\nMerging {len(fragments_files)} fragment parts...")
        df_fragments_list = [pd.read_csv(f) for f in tqdm(fragments_files, desc="fragment parts")]
        df_fragments_merged = pd.concat(df_fragments_list, ignore_index=True)
        
        output_fragments = os.path.join(output_dir, f"{mgf_basename}_fragments.csv")
        df_fragments_merged.to_csv(output_fragments, index=False, encoding='utf-8-sig')
        print(f"  wrote fragments: {output_fragments} ({len(df_fragments_merged)} rows)")
    
    return len(summary_files), len(fragments_files)

def read_formulas_file(formulas_file: str, spectra: List[Dict]) -> Dict[str, str]:
    """CSV (FEATURE_ID + formula or formula only) or TXT one formula per line."""
    formulas_dict = {}
    
    if formulas_file.endswith('.csv'):
        print("Reading formulas CSV...")
        df_csv = pd.read_csv(formulas_file)
        print(f"CSV loaded. Total rows: {len(df_csv)}")
        
        if 'FEATURE_ID' in df_csv.columns and 'formula' in df_csv.columns:
            formulas_dict = dict(zip(df_csv['FEATURE_ID'], df_csv['formula']))
            print(f"Using FEATURE_ID mapping. Total IDs: {len(formulas_dict)}")
        elif 'formula' in df_csv.columns:
            formulas_list = df_csv['formula'].tolist()
            for i, formula in enumerate(formulas_list):
                if i < len(spectra):
                    identifier = spectra[i].get('params', {}).get('IDENTIFIER', 
                                                                  spectra[i].get('params', {}).get('FEATURE_ID', 
                                                                                                  f'Spectrum_{i}'))
                    formulas_dict[identifier] = formula
            print(f"Using row order mapping. Total formulas: {len(formulas_dict)}")
        else:
            raise ValueError("CSV must have a 'formula' column; optional 'FEATURE_ID'")
    
    elif formulas_file.endswith('.txt'):
        print("Reading formulas TXT...")
        with open(formulas_file, 'r', encoding='utf-8') as f:
            formulas_list = [line.strip() for line in f if line.strip()]
        print(f"TXT loaded. Total formulas: {len(formulas_list)}")
        
        for i, formula in enumerate(formulas_list):
            if i < len(spectra):
                identifier = spectra[i].get('params', {}).get('IDENTIFIER', 
                                                              spectra[i].get('params', {}).get('FEATURE_ID', 
                                                                                              f'Spectrum_{i}'))
                formulas_dict[identifier] = formula
        
        print(f"Using row order mapping. Total formulas: {len(formulas_dict)}")
        if len(formulas_list) != len(spectra):
            print(f"WARNING: formulas ({len(formulas_list)}) != spectra ({len(spectra)})")
    else:
        raise ValueError("Formulas file must be .csv or .txt")
    
    return formulas_dict

def parse_arguments():
    """CLI for fragment extraction."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Subformula assignment (assign_subformula) with MS2 preprocess',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
  python subformula.py --mgf file.mgf --formulas formulas.txt --ms2-tol 5.0
        """
    )
    
    parser.add_argument('--mgf', type=str, required=True,
                        help='Input MGF path')
    parser.add_argument('--formulas', type=str, required=True,
                        help='Formulas: CSV or TXT')
    
    parser.add_argument('--msbuddy-dir', type=str, default=None,
                        help='msbuddy-main root (default: auto-detect near this file)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: <mgf_dir>/results)')
    
    parser.add_argument(
        '--ms1-tol',
        type=float,
        default=2.0,
        help='MS1 tolerance in ppm (default 2; informational; assign_subformula uses MS2 tol only)',
    )
    parser.add_argument('--ms2-tol', type=float, default=5.0,
                        help='MS2 tolerance in ppm (default 5)')
    parser.add_argument('--ppm', action='store_true', default=True,
                        help='Tolerance in ppm (default)')
    parser.add_argument('--no-ppm', dest='ppm', action='store_false',
                        help='Tolerance in Da')
    parser.add_argument('--dbe-cutoff', type=float, default=0.0,
                        help='DBE cutoff (default 0)')
    parser.add_argument('--rel-int-cutoff', type=float, default=0.01,
                        help='Relative intensity denoise cutoff (default 0.01)')
    parser.add_argument('--top-n-per-50da', type=int, default=6,
                        help='Max peaks per 50 Da (default 6)')
    parser.add_argument('--remove-isotopes', action='store_true', default=True,
                        help='Remove isotope peaks (default on)')
    parser.add_argument('--no-remove-isotopes', dest='remove_isotopes', action='store_false',
                        help='Keep isotope peaks')
    parser.add_argument('--isotope-tol', type=float, default=0.01,
                        help='Isotope m/z tol Da (default 0.01)')
    parser.add_argument('--isotope-ratio-max', type=float, default=0.5,
                        help='Isotope / monoisotope max intensity ratio (default 0.5)')
    parser.add_argument('--no-preprocess', action='store_true', default=False,
                        help='Skip full preprocess; only strip near precursor (default off)')
    
    parser.add_argument('--n-cores', type=int, default=6,
                        help='Worker processes (default 6)')
    parser.add_argument('--save-interval', type=int, default=30000,
                        help='Save chunk every N spectra (default 30000)')
    
    parser.add_argument('--enable-calibration', action='store_true', default=False,
                        help='Run optional LOWESS calibration on merged fragments CSV')
    parser.add_argument('--lowess-frac', type=float, default=0.1,
                        help='LOWESS frac (default 0.1)')
    parser.add_argument('--error-threshold', type=float, default=5.0,
                        help='Post-calibration |ppm| filter (default 5)')
    
    return parser.parse_args()

if __name__ == "__main__":
    import argparse
    
    args = parse_arguments()
    
    MGF_FILE = os.path.abspath(args.mgf)
    FORMULAS_FILE = os.path.abspath(args.formulas)
    
    if args.msbuddy_dir:
        MSBUDDY_DIR = os.path.abspath(args.msbuddy_dir)
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        msbuddy_candidates = [
            str(Path(__file__).resolve().parents[3] / "src"),
            os.path.join(current_dir, "msbuddy-main"),
            os.path.join(os.path.dirname(current_dir), "msbuddy-main"),
            os.path.join(current_dir, "..", "msbuddy-main"),
        ]
        MSBUDDY_DIR = None
        for candidate in msbuddy_candidates:
            candidate = os.path.abspath(candidate)
            if os.path.exists(candidate) and os.path.isdir(candidate):
                MSBUDDY_DIR = candidate
                break
        
        if MSBUDDY_DIR is None:
            MSBUDDY_DIR = os.path.join(os.path.dirname(os.path.dirname(MGF_FILE)), "msbuddy-main")
            if not os.path.exists(MSBUDDY_DIR):
                raise ValueError("Set --msbuddy-dir to the msbuddy-main root")
    
    if args.output_dir:
        OUTPUT_DIR = os.path.abspath(args.output_dir)
    else:
        OUTPUT_DIR = os.path.join(os.path.dirname(MGF_FILE), "results")
    
    TEMP_DIR = os.path.join(OUTPUT_DIR, "temp")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    NUM_CORES = args.n_cores
    SAVE_INTERVAL = args.save_interval
    
    sys.path.insert(0, MSBUDDY_DIR)
    try:
        from msbuddy.cand import assign_subformula as _assign_subformula
        from msbuddy.base import calc_formula_dbe as _calc_formula_dbe
        from msbuddy.utils import read_formula as _read_formula
    except Exception as e:
        raise ModuleNotFoundError(
            "Cannot import msbuddy. Point --msbuddy-dir at the folder containing the `msbuddy` package (msbuddy-main).\n"
            f"MSBUDDY_DIR={MSBUDDY_DIR}\n"
            f"{type(e).__name__}: {e}"
        )
    globals()["assign_subformula"] = _assign_subformula
    globals()["calc_formula_dbe"] = _calc_formula_dbe
    globals()["read_formula"] = _read_formula
    
    print("=" * 80)
    print("Initialization")
    print("=" * 80)
    print(f"MGF file: {MGF_FILE}")
    print(f"Formulas file: {FORMULAS_FILE}")
    print(f"MSBuddy dir: {MSBUDDY_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    
    print("\nReading MGF file...")
    spectra = read_mgf_manual(MGF_FILE)
    print(f"MGF loaded. Total spectra: {len(spectra)}")
    
    if len(spectra) == 0:
        raise ValueError(f"Empty or unparseable MGF: {MGF_FILE}")
    
    print("\nReading formulas file...")
    formulas_dict = read_formulas_file(FORMULAS_FILE, spectra)
    print(f"Formulas mapping built. Total formulas: {len(formulas_dict)}")
    
    config = {
        'MS1_TOL': args.ms1_tol,
        'MS2_TOL': args.ms2_tol,
        'PPM': args.ppm,
        'DBE_CUTOFF': args.dbe_cutoff,
        'REL_INT_DENOISE_CUTOFF': args.rel_int_cutoff,
        'TOP_N_PER_50_DA': args.top_n_per_50da,
        'REMOVE_ISOTOPES': args.remove_isotopes,
        'ISOTOPE_TOL': args.isotope_tol,
        'ISOTOPE_RATIO_MAX': args.isotope_ratio_max,
        'NO_PREPROCESS': args.no_preprocess
    }
    
    print(f"\nParameters:")
    print(
        f"  MS1 tolerance: {config['MS1_TOL']} {'ppm' if config['PPM'] else 'Da'} "
        f"(informational; assign_subformula uses MS2 tol only)"
    )
    print(f"  Skip preprocess: {config['NO_PREPROCESS']}")
    if config['NO_PREPROCESS']:
        print(f"  NOTE: Preprocessing is skipped; raw MS2 peaks are used (only precursor peak removal).")
        print(f"  MS2 tolerance: {config['MS2_TOL']} {'ppm' if config['PPM'] else 'Da'}")
        print(f"  DBE cutoff: {config['DBE_CUTOFF']}")
    else:
        print(f"  MS2 tolerance: {config['MS2_TOL']} {'ppm' if config['PPM'] else 'Da'}")
        print(f"  DBE cutoff: {config['DBE_CUTOFF']}")
        print(f"  Rel-int denoise cutoff: {config['REL_INT_DENOISE_CUTOFF']}")
        print(f"  Top-N per 50 Da: {config['TOP_N_PER_50_DA']}")
        print(f"  Remove isotopes: {config['REMOVE_ISOTOPES']}")
        if config['REMOVE_ISOTOPES']:
            print(f"    Isotope tol: {config['ISOTOPE_TOL']} Da")
            print(f"    Isotope ratio max: {config['ISOTOPE_RATIO_MAX']}")
    print(f"  CPU cores: {NUM_CORES}")
    print(f"  Save interval: every {SAVE_INTERVAL:,} spectra")
    
    print("\nPreparing tasks...")
    process_args = [(i, spec, formulas_dict, config) for i, spec in enumerate(spectra)]
    
    print("\nProcessing (multi-core)...")
    all_results = []
    all_fragments = []
    fragment_counts = []
    part_num = 0
    
    with Pool(processes=NUM_CORES) as pool:
        results_iter = pool.imap(process_single_spectrum, process_args)
        
        for result in tqdm(results_iter, total=len(spectra), desc="Spectra", unit="sp"):
            if result is not None:
                all_results.append({
                    'spectrum_idx': result['spectrum_idx'],
                    'identifier': result['identifier'],
                    'precursor_mz': result['precursor_mz'],
                    'precursor_formula': result['precursor_formula'],
                    'adduct': result['adduct'],
                    'raw_peaks': result['raw_peaks'],
                    'processed_peaks': result['processed_peaks'],
                    'matched_peaks': result['matched_peaks'],
                    'radical_count': result['radical_count'],
                    'non_radical_count': result['non_radical_count']
                })
                num_fragments = len(result['fragments'])
                fragment_counts.append(num_fragments)
                all_fragments.extend(result['fragments'])
            
            if len(all_results) > 0 and len(all_results) % SAVE_INTERVAL == 0:
                summary_data = []
                start_idx = len(all_results) - SAVE_INTERVAL
                end_idx = len(all_results)
                
                for r in all_results[start_idx:end_idx]:
                    summary_data.append({
                        'Spectrum': r['spectrum_idx'],
                        'ID': r['identifier'],
                        'Precursor m/z': f"{r['precursor_mz']:.6f}",
                        'Formula': r['precursor_formula'],
                        'Raw Peaks': r['raw_peaks'],
                        'Processed Peaks': r['processed_peaks'],
                        'Matched Peaks': r['matched_peaks'],
                        'Radical Ions': r['radical_count'],
                        'Non-Radical Ions': r['non_radical_count']
                    })
                
                fragments_start_idx = sum(fragment_counts[:start_idx]) if start_idx > 0 else 0
                fragments_end_idx = sum(fragment_counts[:end_idx])
                fragments_data = all_fragments[fragments_start_idx:fragments_end_idx]
                
                save_temp_files(part_num, summary_data, fragments_data, TEMP_DIR)
                print(f"\nSaved part {part_num + 1} ({len(summary_data):,} spectra, {len(fragments_data):,} fragments)")
                part_num += 1
    
    remaining_count = len(all_results) % SAVE_INTERVAL
    if remaining_count > 0:
        summary_data = []
        start_idx = len(all_results) - remaining_count
        
        for r in all_results[start_idx:]:
            summary_data.append({
                'Spectrum': r['spectrum_idx'],
                'ID': r['identifier'],
                'Precursor m/z': f"{r['precursor_mz']:.6f}",
                'Formula': r['precursor_formula'],
                'Raw Peaks': r['raw_peaks'],
                'Processed Peaks': r['processed_peaks'],
                'Matched Peaks': r['matched_peaks'],
                'Radical Ions': r['radical_count'],
                'Non-Radical Ions': r['non_radical_count']
            })
        
        fragments_start_idx = sum(fragment_counts[:start_idx]) if start_idx > 0 else 0
        fragments_data = all_fragments[fragments_start_idx:]
        save_temp_files(part_num, summary_data, fragments_data, TEMP_DIR)
        print(f"\nSaved last part ({len(summary_data):,} spectra, {len(fragments_data):,} fragments)")
        part_num += 1
    
    print("\n" + "=" * 80)
    print("Final summary")
    print("=" * 80)
    
    print(f"spectra: {len(spectra)}")
    print(f"ok:      {len(all_results)}")
    print(f"skipped: {len(spectra) - len(all_results)}")
    print(f"fragment rows: {len(all_fragments)}")
    print(f"temp parts: {part_num}")
    
    if len(all_results) > 0:
        total_matched = sum(r['matched_peaks'] for r in all_results)
        total_radical = sum(r['radical_count'] for r in all_results)
        total_non_radical = sum(r['non_radical_count'] for r in all_results)
        
        print(f"\nIon counts (matched):")
        print(f"  total matched: {total_matched}")
        print(f"  radical: {total_radical} ({total_radical/total_matched*100:.1f}%)" if total_matched > 0 else "  radical: 0")
        print(f"  non-radical: {total_non_radical} ({total_non_radical/total_matched*100:.1f}%)" if total_matched > 0 else "  non-radical: 0")
    
    print("\n" + "=" * 80)
    print("Merging temp files")
    print("=" * 80)
    
    num_summary_parts, num_fragments_parts = merge_temp_files(TEMP_DIR, OUTPUT_DIR, MGF_FILE)
    
    if args.enable_calibration:
        print("\n" + "=" * 80)
        print("Optional calibration")
        print("=" * 80)
        
        mgf_basename = os.path.splitext(os.path.basename(MGF_FILE))[0]
        fragments_file = os.path.join(OUTPUT_DIR, f"{mgf_basename}_fragments.csv")
        
        if not os.path.exists(fragments_file):
            print(f"  error: missing {fragments_file}")
        else:
            print(f"\nFragments: {fragments_file}")
            df_fragments = pd.read_csv(fragments_file)
            print(f"  rows: {len(df_fragments):,}")
            
            df_fragments['Mass_Error_ppm'] = pd.to_numeric(df_fragments['Mass_Error_ppm'], errors='coerce')
            df_fragments['m/z'] = pd.to_numeric(df_fragments['m/z'], errors='coerce')
            
            print("\n[1] Outlier screen...")
            df_clean = dynamic_outlier_removal(df_fragments, frac=args.lowess_frac)
            
            print("\n[2] LOWESS...")
            lowess_fit, mz_values, df_sorted = lowess_calibration(df_clean, frac=args.lowess_frac)
            
            if lowess_fit is not None:
                print("\n[3] Apply calibration...")
                df_calibrated = apply_calibration_to_fragments(df_clean, lowess_fit, mz_values)
                
                print("\n[4] Error filter...")
                df_final = filter_fragments_by_error(df_calibrated, threshold=args.error_threshold)
                
                print("\n[5] Before/after table...")
                comparison_df = df_final.copy()
                
                comparison_df['Mass_Error_ppm_Original'] = comparison_df['Mass_Error_ppm']
                comparison_df['Mass_Error_ppm_Corrected'] = comparison_df['Mass_Error_ppm_corrected']
                comparison_df['m/z_Original'] = comparison_df['m/z']
                comparison_df['m/z_Corrected'] = comparison_df['m/z_corrected']
                
                mask_valid = comparison_df['Mass_Error_ppm_Corrected'].notna()
                comparison_df['Error_Improvement_ppm'] = np.nan
                comparison_df['Abs_Error_Original'] = np.nan
                comparison_df['Abs_Error_Corrected'] = np.nan
                comparison_df['Abs_Error_Improvement'] = np.nan
                
                if mask_valid.sum() > 0:
                    comparison_df.loc[mask_valid, 'Error_Improvement_ppm'] = (
                        comparison_df.loc[mask_valid, 'Mass_Error_ppm_Original'] - 
                        comparison_df.loc[mask_valid, 'Mass_Error_ppm_Corrected']
                    )
                    comparison_df.loc[mask_valid, 'Abs_Error_Original'] = comparison_df.loc[mask_valid, 'Mass_Error_ppm_Original'].abs()
                    comparison_df.loc[mask_valid, 'Abs_Error_Corrected'] = comparison_df.loc[mask_valid, 'Mass_Error_ppm_Corrected'].abs()
                    comparison_df.loc[mask_valid, 'Abs_Error_Improvement'] = (
                        comparison_df.loc[mask_valid, 'Abs_Error_Original'] - 
                        comparison_df.loc[mask_valid, 'Abs_Error_Corrected']
                    )
                
                output_columns = [
                    'Spectrum', 'Identifier', 'Precursor Formula', 'Precursor m/z', 'Adduct',
                    'Peak Index', 'Formula',
                    'm/z_Original', 'm/z_Corrected', 'm/z_Shift',
                    'Mass_Error_ppm_Original', 'Mass_Error_ppm_Corrected', 'Error_Improvement_ppm',
                    'Abs_Error_Original', 'Abs_Error_Corrected', 'Abs_Error_Improvement',
                    'LOWESS_delta_ppm', 'Num Candidates', 'DBE', 'Is Radical'
                ]
                
                comparison_df['m/z_Shift'] = np.nan
                mask_valid = comparison_df['m/z_Corrected'].notna()
                if mask_valid.sum() > 0:
                    comparison_df.loc[mask_valid, 'm/z_Shift'] = (
                        comparison_df.loc[mask_valid, 'm/z_Corrected'] - 
                        comparison_df.loc[mask_valid, 'm/z_Original']
                    )
                
                available_columns = [col for col in output_columns if col in comparison_df.columns]
                comparison_output = comparison_df[available_columns].copy()
                
                comparison_file = os.path.join(OUTPUT_DIR, f"{mgf_basename}_fragments_calibration_comparison.csv")
                comparison_output.to_csv(comparison_file, index=False, encoding='utf-8-sig')
                print(f"  wrote comparison: {comparison_file} ({len(comparison_output):,} rows)")
                
                valid_comparison = comparison_output[comparison_output['Mass_Error_ppm_Corrected'].notna()]
                if len(valid_comparison) > 0:
                    print(f"\n  Before/after |ppm| stats:")
                    print(f"    mean |orig|: {valid_comparison['Abs_Error_Original'].mean():.3f} ppm")
                    print(f"    mean |corr|: {valid_comparison['Abs_Error_Corrected'].mean():.3f} ppm")
                    print(f"    mean improve: {valid_comparison['Abs_Error_Improvement'].mean():.3f} ppm")
                    print(f"    improved: {(valid_comparison['Abs_Error_Improvement'] > 0).sum():,} ({(valid_comparison['Abs_Error_Improvement'] > 0).sum()/len(valid_comparison)*100:.2f}%)")
                    print(f"    worse:    {(valid_comparison['Abs_Error_Improvement'] < 0).sum():,} ({(valid_comparison['Abs_Error_Improvement'] < 0).sum()/len(valid_comparison)*100:.2f}%)")
                
                calibrated_fragments_file = os.path.join(OUTPUT_DIR, f"{mgf_basename}_fragments_calibrated.csv")
                df_final.to_csv(calibrated_fragments_file, index=False, encoding='utf-8-sig')
                print(f"\n  wrote calibrated: {calibrated_fragments_file}")
            else:
                print("  warn: LOWESS failed; skip rest")
    
    import glob
    temp_files = glob.glob(os.path.join(TEMP_DIR, "*.csv"))
    for f in temp_files:
        os.remove(f)
    print(f"\nDeleted {len(temp_files)} temp files")
    
    print("\n" + "=" * 80)
    print("Done.")
    print("=" * 80)
