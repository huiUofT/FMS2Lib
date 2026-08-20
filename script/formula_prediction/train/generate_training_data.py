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

Build Msbuddy ML training matrices from MGF + formulas (batched, parallel).
"""

import numpy as np
import joblib
from pathlib import Path
import sys
import os
import argparse
from scipy.stats import norm
from tqdm import tqdm
import shutil
import math
import time
from joblib import Parallel, delayed


def _prepend_msbuddy_on_path() -> None:
    """Prefer ``<repo>/src/msbuddy``; else a parent directory containing ``msbuddy/``."""
    for anc in Path(__file__).resolve().parents:
        if (anc / "src" / "msbuddy").is_dir():
            s = str(anc / "src")
            if s not in sys.path:
                sys.path.insert(0, s)
            return
        if (anc / "msbuddy" / "__init__.py").is_file():
            s = str(anc)
            if s not in sys.path:
                sys.path.insert(0, s)
            return


_prepend_msbuddy_on_path()

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from msbuddy.main import Msbuddy, MsbuddyConfig, _gen_subformula
from msbuddy.load import init_db
from msbuddy.utils import read_formula, form_arr_to_str
from msbuddy.base import Formula, CandidateFormula
from msbuddy.ml import gen_ml_feature_single, _gen_form_feature
from msbuddy.cand import _calc_ms1_iso_sim

# Returned from process_one_spectrum when --no-insert-ground-truth and GT formula
# matches no MsBuddy candidate (skip spectrum; no LambdaRank-positive row).
_NO_POSITIVE_LABEL = object()


def process_one_spectrum(msb,
                         idx,
                         gt_form_arr,
                         ms1_tol,
                         ms2_tol,
                         random_seed,
                         shared_data_dict,
                         max_candidates_per_spectrum,
                         global_idx,
                         insert_ground_truth=True):
    """
    Per-spectrum: optional cap on candidates, formula features, optional prepend of
    ground-truth candidate, subformula assignment, MS1 isotope sim, ML features
    (m/z shift augment on true-labeled rows only).
    Returns None on failure; _NO_POSITIVE_LABEL if GT not among candidates (no-insert mode);
    else (X_arr_one, y_arr_one, group_size, formula_strs_list, gt_form_str).
    """
    try:
        mf = msb.data[idx]
        if mf is None:
            return None
        if not mf.candidate_formula_list:
            return None

        gt_form_str = form_arr_to_str(gt_form_arr)

        if max_candidates_per_spectrum is not None and \
           len(mf.candidate_formula_list) > max_candidates_per_spectrum:
            mf.candidate_formula_list = mf.candidate_formula_list[:max_candidates_per_spectrum]

        all_form_arrs = []
        all_dbe_arrs = []
        all_mass_arrs = []
        for cf in mf.candidate_formula_list:
            all_form_arrs.append(cf.formula.array)
            all_dbe_arrs.append(cf.formula.dbe)
            all_mass_arrs.append(cf.formula.mass)

        feature_arr = _gen_form_feature(
            np.array(all_form_arrs),
            np.array(all_dbe_arrs),
            np.array(all_mass_arrs)
        )
        for j, cf in enumerate(mf.candidate_formula_list):
            cf.formula_feature_array = feature_arr[j]

        if insert_ground_truth:
            form = Formula(gt_form_arr, 0)
            true_cf = CandidateFormula(
                formula=form,
                charged_formula=Formula(gt_form_arr + mf.adduct.net_formula.array,
                                        mf.adduct.charge)
            )
            true_cf.formula_feature_array = _gen_form_feature(
                np.array([gt_form_arr]),
                np.array([form.dbe]),
                np.array([form.mass])
            )[0]

            if mf.ms1_processed and mf.ms1_processed.int_array is not None:
                true_cf.ms1_isotope_similarity = _calc_ms1_iso_sim(true_cf, mf, 4)
            else:
                true_cf.ms1_isotope_similarity = 0.0

            new_cand_list = [true_cf]
            for cf in mf.candidate_formula_list:
                if form_arr_to_str(cf.formula.array) != gt_form_str:
                    new_cand_list.append(cf)

            mf.candidate_formula_list = new_cand_list

        mf = _gen_subformula(mf, msb.config)

        X_list = []
        y_list = []
        formula_strs_list = []
        found_positive = False

        for n, cf in enumerate(mf.candidate_formula_list):
            if mf.ms1_processed and mf.ms1_processed.int_array is not None:
                cf.ms1_isotope_similarity = _calc_ms1_iso_sim(cf, mf, 4)
            else:
                cf.ms1_isotope_similarity = 0.0

            if insert_ground_truth:
                this_true = (n == 0)
            else:
                cand_str = form_arr_to_str(cf.formula.array)
                this_true = (not found_positive) and (cand_str == gt_form_str)
                if this_true:
                    found_positive = True

            feat = gen_ml_feature_single(
                mf, cf, True, ms1_tol, ms2_tol, shared_data_dict
            )

            if this_true:
                mz_shift = np.random.normal(0, ms1_tol / 5)
                mz_p = norm.cdf(mz_shift, loc=0, scale=ms1_tol / 3)
                mz_p = mz_p if mz_p < 0.5 else 1 - mz_p
                log_p = np.log(mz_p * 2)
                feat[1] = np.clip(log_p, -2, 0)

            X_list.append(feat)
            y_list.append(1 if this_true else 0)
            formula_strs_list.append(form_arr_to_str(cf.formula.array))

        if not X_list:
            return None

        if not insert_ground_truth and not found_positive:
            return _NO_POSITIVE_LABEL

        X_one = np.vstack(X_list)
        y_one = np.array(y_list)
        group_size = len(y_list)

        return X_one, y_one, group_size, formula_strs_list, gt_form_str

    except Exception as e:
        if global_idx < 3:
            print(f"  WARN sample {global_idx}: {type(e).__name__}: {str(e)}")
        return None


def process_batch_parallel(msb,
                           valid_indices_batch,
                           gt_form_arrs_batch,
                           batch_idx,
                           total_batches,
                           ms1_tol,
                           ms2_tol,
                           random_seed,
                           shared_data_dict,
                           global_start_idx=0,
                           max_candidates_per_spectrum=None,
                           n_jobs=1,
                           insert_ground_truth=True):
    """Process one batch via joblib threading; returns batch arrays and stats dict."""
    print(f"  [batch {batch_idx + 1}/{total_batches}] threads={n_jobs}, spectra={len(valid_indices_batch)}")

    tasks = []
    for local_i, (idx, gt_form_arr) in enumerate(
            zip(valid_indices_batch, gt_form_arrs_batch)):
        global_idx = global_start_idx + local_i
        tasks.append((idx, gt_form_arr, global_idx))

    results = Parallel(
        n_jobs=n_jobs,
        backend="threading"
    )(
        delayed(process_one_spectrum)(
            msb=msb,
            idx=idx,
            gt_form_arr=gt_form_arr,
            ms1_tol=ms1_tol,
            ms2_tol=ms2_tol,
            random_seed=random_seed,
            shared_data_dict=shared_data_dict,
            max_candidates_per_spectrum=max_candidates_per_spectrum,
            global_idx=global_idx,
            insert_ground_truth=insert_ground_truth,
        )
        for (idx, gt_form_arr, global_idx) in tasks
    )

    X_list = []
    y_list = []
    group_list = []
    formula_strs_list = []
    gt_formula_per_group_batch = []

    processed_count = 0
    skipped_count = 0
    skipped_no_positive_count = 0
    feature_calc_failed_count = 0
    no_candidates_count = 0

    for res in results:
        if res is _NO_POSITIVE_LABEL:
            skipped_no_positive_count += 1
            continue
        if res is None:
            skipped_count += 1
            continue
        X_one, y_one, group_size, formula_strs, gt_form_str = res
        X_list.append(X_one)
        y_list.append(y_one)
        group_list.append(group_size)
        formula_strs_list.extend(formula_strs)
        gt_formula_per_group_batch.append(gt_form_str)
        processed_count += 1

    if not X_list:
        X_arr_batch = np.array([])
        y_arr_batch = np.array([])
        group_arr_batch = np.array([], dtype=int)
        formula_strs_batch = []
    else:
        X_arr_batch = np.vstack(X_list)
        y_arr_batch = np.concatenate(y_list)
        group_arr_batch = np.array(group_list, dtype=int)
        formula_strs_batch = formula_strs_list

    stats = {
        "processed_count": processed_count,
        "skipped_count": skipped_count,
        "skipped_no_positive_count": skipped_no_positive_count,
        "feature_calc_failed_count": feature_calc_failed_count,
        "no_candidates_count": no_candidates_count,
    }

    print(f"  [batch {batch_idx + 1}/{total_batches}] done: ok={processed_count}, skipped={skipped_count}, "
          f"no_positive_in_candidates={skipped_no_positive_count}, "
          f"n_rows={len(y_arr_batch)}, n_groups={len(group_arr_batch)}")

    return (
        X_arr_batch,
        y_arr_batch,
        group_arr_batch,
        formula_strs_batch,
        gt_formula_per_group_batch,
        stats,
    )


def prepare_training_data(mgf_path,
                           formulas_path,
                           output_dir="training_data",
                           ms1_tol=2.0,
                           ms2_tol=5.0,
                           random_seed=24,
                           n_cpu=-1,
                           batch_size=20000,
                           max_candidates_per_spectrum=None,
                           insert_ground_truth=True):
    """Load MGF + formulas, generate features in batches; return success bool."""
    start_time = time.time()

    print("=" * 80)
    print("Msbuddy training data (generate_training_data)")
    print("=" * 80)
    print(
        "Insert ground-truth candidate at rank 0: "
        f"{'yes' if insert_ground_truth else 'no (label by matching formula to candidates)'}"
    )

    print("\n" + "=" * 80)
    print("Step 1: load MGF")
    print("=" * 80)

    mgf_path = os.path.abspath(mgf_path)
    print(f"cwd: {os.getcwd()}")
    print(f"MGF: {mgf_path}")

    if not os.path.exists(mgf_path):
        print(f"ERROR: MGF not found: {mgf_path}")
        return False

    print("Loading MGF via Msbuddy...")
    config = MsbuddyConfig(
        ms_instr="orbitrap",
        ppm=True,
        ms1_tol=ms1_tol,
        ms2_tol=ms2_tol,
        halogen=True,
        parallel=True,
        n_cpu=n_cpu,
    )
    # Orbitrap preset may override tolerances; force user values
    config.ms1_tol = ms1_tol
    config.ms2_tol = ms2_tol
    print(f"Tolerances: requested ms1={ms1_tol}, ms2={ms2_tol}; config ms1={config.ms1_tol}, ms2={config.ms2_tol}")
    msb = Msbuddy(config)
    msb.load_mgf(mgf_path)
    print(f"Spectra loaded: {len(msb.data)}")

    if n_cpu == -1:
        print("Msbuddy internal parallel: all CPUs")
        n_jobs = os.cpu_count() or 1
    else:
        print(f"Msbuddy internal parallel: n_cpu={msb.config.n_cpu}")
        n_jobs = n_cpu

    print("\nFirst 3 spectra:")
    for i in range(min(3, len(msb.data))):
        mf = msb.data[i]
        print(f"  [{i}] id={mf.identifier}, m/z={mf.mz:.6f}, adduct={mf.adduct}")

    print("\n" + "=" * 80)
    print("Step 2: read formulas (one per line)")
    print("=" * 80)

    formulas_path = os.path.abspath(formulas_path)
    print(f"Formulas file: {formulas_path}")

    if not os.path.exists(formulas_path):
        print(f"ERROR: formulas file not found: {formulas_path}")
        return False

    with open(formulas_path, "r") as f:
        formula_list = [line.strip() for line in f if line.strip()]

    print(f"Formulas read: {len(formula_list)}")

    if len(msb.data) != len(formula_list):
        min_len = min(len(msb.data), len(formula_list))
        print(f"WARN: n_spectra={len(msb.data)} != n_formulas={len(formula_list)}; using first {min_len}")
    else:
        min_len = len(msb.data)
        print(f"Spectra and formulas aligned: {min_len}")

    print("\n" + "=" * 80)
    print("Step 3: parse formulas")
    print("=" * 80)

    gt_form_arrs = []
    valid_indices = []
    invalid_count = 0

    for i, formula_str in enumerate(formula_list[:min_len]):
        form_arr = read_formula(formula_str)
        if form_arr is None:
            print(f"WARN: invalid formula at index {i}: {formula_str!r} (skip)")
            invalid_count += 1
            continue
        gt_form_arrs.append(form_arr)
        valid_indices.append(i)
        if i < 5:
            print(f"  {formula_str} -> {form_arr}")

    print(f"Valid formulas: {len(gt_form_arrs)}")
    if invalid_count > 0:
        print(f"Skipped invalid formulas: {invalid_count}")

    print("\n" + "=" * 80)
    print("Step 4: candidate generation (Msbuddy)")
    print("=" * 80)

    print("Generating candidates for all spectra (may be slow)...")
    msb._preprocess_and_generate_candidate_formula(0, len(msb.data))
    print("Candidate generation done.")

    total_candidates = sum(
        len(mf.candidate_formula_list)
        for mf in msb.data
        if mf.candidate_formula_list
    )
    valid_mf_count = sum(
        1 for mf in msb.data if mf.candidate_formula_list
    )
    avg_candidates = total_candidates / valid_mf_count if valid_mf_count > 0 else 0.0

    print(f"Total candidates: {total_candidates}")
    print(f"Spectra with candidates: {valid_mf_count}")
    print(f"Avg candidates per spectrum: {avg_candidates:.1f}")

    if max_candidates_per_spectrum is not None:
        print(f"\nCapping candidates to {max_candidates_per_spectrum} per spectrum...")
        limited_count = 0
        for mf in msb.data:
            if mf.candidate_formula_list and len(mf.candidate_formula_list) > max_candidates_per_spectrum:
                mf.candidate_formula_list = mf.candidate_formula_list[:max_candidates_per_spectrum]
                limited_count += 1

        total_candidates_after = sum(
            len(mf.candidate_formula_list)
            for mf in msb.data
            if mf.candidate_formula_list
        )
        avg_candidates_after = total_candidates_after / valid_mf_count if valid_mf_count > 0 else 0.0

        print(f"Capped {limited_count} spectra")
        print(f"Total candidates after cap: {total_candidates_after}")
        print(f"Avg candidates after cap: {avg_candidates_after:.1f}")
    else:
        print("No per-spectrum candidate cap")

    print("\n" + "=" * 80)
    print("Step 5: init Msbuddy DB (fragments / losses / formulas)")
    print("=" * 80)

    shared_data_dict = init_db()
    print("init_db done")

    print("\n" + "=" * 80)
    print("Step 6: batched parallel feature extraction")
    print("=" * 80)

    total_samples = len(valid_indices)
    if total_samples == 0:
        print("ERROR: no valid samples")
        return False

    n_batches = math.ceil(total_samples / batch_size)

    print(f"Valid spectra (with parsed formula): {total_samples}")
    print(f"Batch size: {batch_size}")
    print(f"Batches: {n_batches}")
    print(f"ML feature threads (joblib): {n_jobs}")
    print(f"MS1 tolerance: {ms1_tol} ppm")
    print(f"MS2 tolerance: {ms2_tol} ppm")

    output_dir_path = Path(output_dir)
    temp_dir = output_dir_path / "temp_batches"
    temp_dir.mkdir(parents=True, exist_ok=True)

    total_processed_count = 0
    total_skipped_count = 0
    total_feature_calc_failed_count = 0
    total_no_candidates_count = 0

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_samples)

        print(f"\nBatch {batch_idx + 1}/{n_batches} (sample index {start_idx}..{end_idx - 1})")

        valid_indices_batch = valid_indices[start_idx:end_idx]
        gt_form_arrs_batch = [gt_form_arrs[i] for i in range(start_idx, end_idx)]

        X_arr_batch, y_arr_batch, group_arr_batch, formula_strs_batch, _gt_fg_batch, stats = process_batch_parallel(
            msb=msb,
            valid_indices_batch=valid_indices_batch,
            gt_form_arrs_batch=gt_form_arrs_batch,
            batch_idx=batch_idx,
            total_batches=n_batches,
            ms1_tol=ms1_tol,
            ms2_tol=ms2_tol,
            random_seed=random_seed,
            shared_data_dict=shared_data_dict,
            global_start_idx=start_idx,
            max_candidates_per_spectrum=max_candidates_per_spectrum,
            n_jobs=n_jobs,
        )

        total_processed_count += stats["processed_count"]
        total_skipped_count += stats["skipped_count"]
        total_feature_calc_failed_count += stats["feature_calc_failed_count"]
        total_no_candidates_count += stats["no_candidates_count"]

        if X_arr_batch.size > 0:
            batch_file = temp_dir / f"batch_{batch_idx:04d}.joblib"
            joblib.dump(
                {
                    "X_arr": X_arr_batch,
                    "y_arr": y_arr_batch,
                    "group_arr": group_arr_batch,
                    "formula_strs": formula_strs_batch,
                },
                batch_file,
            )
            print(f"  saved {batch_file} (rows={len(y_arr_batch)}, groups={len(group_arr_batch)})")
        else:
            print(f"  WARN batch {batch_idx + 1}: no data, skip save")

        del X_arr_batch, y_arr_batch, group_arr_batch, formula_strs_batch

    print("\nBatch summary:")
    print(f"  valid spectra: {total_samples}")
    print(f"  processed: {total_processed_count}")
    print(f"  skipped: {total_skipped_count}")
    print(f"    no candidates (placeholder): {total_no_candidates_count}")
    print(f"    feature fail (placeholder): {total_feature_calc_failed_count}")

    print("\n" + "=" * 80)
    print("Step 7: merge batches")
    print("=" * 80)

    X_arr_list = []
    y_arr_list = []
    group_arr_list = []
    formula_strs_arr = []

    for batch_idx in tqdm(range(n_batches), desc="merge"):
        batch_file = temp_dir / f"batch_{batch_idx:04d}.joblib"
        if batch_file.exists():
            batch_data = joblib.load(batch_file)
            if batch_data["X_arr"].size > 0:
                X_arr_list.append(batch_data["X_arr"])
                y_arr_list.append(batch_data["y_arr"])
                group_arr_list.append(batch_data["group_arr"])
                formula_strs_arr.extend(batch_data["formula_strs"])

    if not X_arr_list:
        print("ERROR: nothing to merge")
        return False

    print("Concatenating arrays...")
    X_arr = np.vstack(X_arr_list)
    y_arr = np.concatenate(y_arr_list)
    group_arr = np.concatenate(group_arr_list)

    print("Merge done")
    print(f"  X shape: {X_arr.shape}")
    print(f"  y length: {len(y_arr)}")
    print(f"  group length: {len(group_arr)}")
    print(f"  positives: {np.sum(y_arr)}")
    print(f"  formula_strs: {len(formula_strs_arr)}")

    print("Removing temp_batches...")
    shutil.rmtree(temp_dir, ignore_errors=True)
    print("Temp dir removed")

    print("\n" + "=" * 80)
    print("Step 8: round X to 8 decimals")
    print("=" * 80)

    X_arr = np.round(X_arr, 8)
    print(f"Done. X shape: {X_arr.shape}")

    print("\n" + "=" * 80)
    print("Step 9: fill MS1 feature column 0 if <= 0")
    print("=" * 80)

    if len(X_arr) > 0:
        need_fill = np.sum(X_arr[:, 0] <= 0)
        if need_fill > 0:
            print(f"Rows needing MS1 column fill: {need_fill}")
            np.random.seed(random_seed)
            X_arr[:, 0] = np.where(
                X_arr[:, 0] <= 0,
                np.random.uniform(0, 1),
                X_arr[:, 0],
            )
            print("MS1 column filled")
        else:
            print("All MS1 values > 0; no fill")
    else:
        print("WARN: X_arr empty; skip fill")

    print("\n" + "=" * 80)
    print("Step 10: save outputs")
    print("=" * 80)

    output_dir_path.mkdir(parents=True, exist_ok=True)

    X_path = output_dir_path / "X_arr.joblib"
    y_path = output_dir_path / "y_arr.joblib"
    g_path = output_dir_path / "group_arr.joblib"
    formula_strs_path = output_dir_path / "formula_strs.joblib"

    joblib.dump(X_arr, X_path)
    joblib.dump(y_arr, y_path)
    joblib.dump(group_arr, g_path)
    joblib.dump(formula_strs_arr, formula_strs_path)

    print(f"Wrote {X_path}")
    print(f"Wrote {y_path}")
    print(f"Wrote {g_path}")
    print(f"Wrote {formula_strs_path}")

    summary_path = output_dir_path / "summary.log"
    end_time = time.time()
    elapsed = end_time - start_time

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Msbuddy training data summary (generate_training_data)\n")
        f.write("=" * 60 + "\n")
        f.write(f"MGF: {mgf_path}\n")
        f.write(f"Formulas file: {formulas_path}\n")
        f.write(f"Output directory: {output_dir_path}\n\n")
        f.write(f"Valid spectra: {total_samples}\n")
        f.write(f"Processed spectra: {total_processed_count}\n")
        f.write(f"Skipped spectra: {total_skipped_count}\n")
        f.write(f"  - no candidates (placeholder stat): {total_no_candidates_count}\n")
        f.write(f"  - feature fail (placeholder stat): {total_feature_calc_failed_count}\n\n")
        f.write(f"Shapes: X={X_arr.shape}, y={y_arr.shape}, groups={len(group_arr)}, formula_strs={len(formula_strs_arr)}\n")
        f.write(f"Positive labels: {int(np.sum(y_arr))}\n")
        f.write(f"Negative labels: {int(len(y_arr) - np.sum(y_arr))}\n\n")
        f.write(f"MS1 tolerance (ppm): {ms1_tol}\n")
        f.write(f"MS2 tolerance (ppm): {ms2_tol}\n")
        f.write(f"Msbuddy n_cpu: {n_cpu}\n")
        f.write(f"joblib threads (feature step): {n_jobs}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(
            f"Max candidates per spectrum: "
            f"{max_candidates_per_spectrum if max_candidates_per_spectrum is not None else 'unlimited'}\n\n"
        )
        f.write(f"Elapsed (minutes): {elapsed/60:.2f}\n")

    print("\n" + "=" * 80)
    print("Done.")
    print("=" * 80)
    print(f"Output directory: {output_dir_path}")
    print(f"Summary log: {summary_path}")
    print(f"Elapsed (minutes): {elapsed/60:.2f}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Build Msbuddy training data from MGF + formula list (batched, parallel)."
    )

    parser.add_argument(
        "--mgf",
        type=str,
        required=True,
        help="Path to MGF file",
    )

    parser.add_argument(
        "--formulas",
        type=str,
        required=True,
        help="Path to formulas text file (one formula per line, same order as MGF)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="training_data",
        help="Output directory (default: training_data)",
    )

    parser.add_argument(
        "--ms1-tol",
        type=float,
        default=2.0,
        help="MS1 tolerance in ppm (default: 2.0)",
    )

    parser.add_argument(
        "--ms2-tol",
        type=float,
        default=5.0,
        help="MS2 tolerance in ppm (default: 5.0)",
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=24,
        help="Random seed (default: 24)",
    )

    parser.add_argument(
        "--n-cpu",
        type=int,
        default=-1,
        help="Msbuddy internal parallel CPU count; -1 = all cores (default: -1)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=20000,
        help="Spectra per batch (default: 20000)",
    )

    parser.add_argument(
        "--max-candidates-per-spectrum",
        type=int,
        default=None,
        help="Cap candidates per spectrum (default: no cap)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("Arguments")
    print("=" * 80)
    print(f"MGF: {args.mgf}")
    print(f"Formulas: {args.formulas}")
    print(f"Output directory: {args.output_dir}")
    print(f"MS1 tolerance (ppm): {args.ms1_tol}")
    print(f"MS2 tolerance (ppm): {args.ms2_tol}")
    print(f"Random seed: {args.random_seed}")
    print(f"Msbuddy n_cpu: {args.n_cpu}")
    print(f"Batch size: {args.batch_size}")
    print(
        f"Max candidates per spectrum: "
        f"{args.max_candidates_per_spectrum if args.max_candidates_per_spectrum is not None else 'unlimited'}"
    )
    print("=" * 80)

    success = prepare_training_data(
        mgf_path=args.mgf,
        formulas_path=args.formulas,
        output_dir=args.output_dir,
        ms1_tol=args.ms1_tol,
        ms2_tol=args.ms2_tol,
        random_seed=args.random_seed,
        n_cpu=args.n_cpu,
        batch_size=args.batch_size,
        max_candidates_per_spectrum=args.max_candidates_per_spectrum,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
