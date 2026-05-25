#!/usr/bin/env python3
"""
Train a single LightGBM LambdaRank model on combined MS1+MS2 (+ formula) features
(columns 1..52 of X_arr from generate_training_data).

Feature layout (same as notebook 20260417): skip index 0 (MS1 isotope sim);
cols 1-2 MS1 misc, 3-28 formula (26-d), 29-52 MS2 (24-d) -> 52-d total.
"""

from __future__ import annotations

import os

# Before NumPy / LightGBM (OpenMP): avoid thread oversubscription SIGSEGV on WSL/Linux.
for _key, _val in (
    ("OMP_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("VECLIB_MAXIMUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
):
    os.environ.setdefault(_key, _val)

import argparse
import faulthandler
import inspect
import re
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from lightgbm import LGBMRanker


def _joblib_load(path: Path, mmap: bool):
    """Load joblib; optional read-only mmap for large arrays (fewer memcpy issues)."""
    try:
        if mmap:
            return joblib.load(path, mmap_mode="r")
    except TypeError:
        pass
    return joblib.load(path)


def _as_writeable_array(a, dtype=np.float32):
    """Dense C-contiguous array for downstream C extensions."""
    if isinstance(a, np.ndarray):
        return np.ascontiguousarray(np.asarray(a, dtype=dtype))
    return np.ascontiguousarray(np.asarray(a, dtype=dtype))


def _lgb_predict(ranker: LGBMRanker, X: np.ndarray) -> np.ndarray:
    """Predict; avoid sklearn feature-name warning when X is ndarray."""
    kwargs = {}
    try:
        if "validate_features" in inspect.signature(ranker.predict).parameters:
            kwargs["validate_features"] = False
    except (TypeError, ValueError):
        pass
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Found 'ndcg_eval_at' in params",
            category=UserWarning,
            module=r"lightgbm\.sklearn",
        )
        warnings.filterwarnings(
            "ignore",
            message=r"does not have valid feature names",
            category=UserWarning,
        )
        return ranker.predict(X, **kwargs)


def extract_f_count(formula_str: str) -> int:
    m = re.search(r"F(\d+)", formula_str)
    if m:
        return int(m.group(1))
    if "F" in formula_str:
        return 1
    return 0


def stratified_split_by_formula(
    X_arr,
    y_arr,
    group_arr,
    true_formulas_per_group,
    f_count_arr,
    val_size=0.1,
    test_size=0.1,
    f_strata=None,
    random_state=24,
):
    """Stratified by F-count strata; splits by formula so train/val/test formulas do not overlap."""
    cumulative_sizes = np.cumsum(group_arr)
    group_indices = np.zeros((len(group_arr), 2), dtype=int)
    group_indices[1:, 0] = cumulative_sizes[:-1]
    group_indices[:, 1] = cumulative_sizes

    if f_strata is None:
        f_strata = [(0, 0), (1, 1), (2, 2), (3, 3), (4, None)]

    group_strata = np.zeros(len(group_arr), dtype=int)
    for stratum_idx, (f_min, f_max) in enumerate(f_strata):
        mask = (
            (f_count_arr >= f_min) if f_max is None else
            ((f_count_arr >= f_min) & (f_count_arr <= f_max))
        )
        group_strata[mask] = stratum_idx

    np.random.seed(random_state)

    X_train, X_val, X_test = [], [], []
    y_train, y_val, y_test = [], [], []
    groups_train, groups_val, groups_test = [], [], []
    f_train, f_val, f_test = [], [], []

    train_group_ids, val_group_ids, test_group_ids = [], [], []
    train_formulas_set = set()
    val_formulas_set = set()
    test_formulas_set = set()

    for stratum_idx, (_f_min, _f_max) in enumerate(f_strata):
        stratum_mask = group_strata == stratum_idx
        stratum_group_indices = np.where(stratum_mask)[0]
        if len(stratum_group_indices) == 0:
            continue

        formula_to_groups = defaultdict(list)
        for group_idx in stratum_group_indices:
            formula = true_formulas_per_group[group_idx]
            formula_to_groups[formula].append(group_idx)

        unique_formulas = list(formula_to_groups.keys())
        np.random.shuffle(unique_formulas)

        n_formulas = len(unique_formulas)
        n_val_formulas = max(1, int(n_formulas * val_size))
        n_test_formulas = max(1, int(n_formulas * test_size))
        n_train_formulas = n_formulas - n_val_formulas - n_test_formulas

        train_formulas = unique_formulas[:n_train_formulas]
        val_formulas = unique_formulas[
            n_train_formulas:n_train_formulas + n_val_formulas
        ]
        test_formulas = unique_formulas[n_train_formulas + n_val_formulas:]

        train_formulas_set.update(train_formulas)
        val_formulas_set.update(val_formulas)
        test_formulas_set.update(test_formulas)

        for formula in train_formulas:
            for group_idx in formula_to_groups[formula]:
                s, e = group_indices[group_idx]
                X_train.extend(X_arr[s:e])
                y_train.extend(y_arr[s:e])
                groups_train.append(group_arr[group_idx])
                f_train.append(f_count_arr[group_idx])
                train_group_ids.append(group_idx)

        for formula in val_formulas:
            for group_idx in formula_to_groups[formula]:
                s, e = group_indices[group_idx]
                X_val.extend(X_arr[s:e])
                y_val.extend(y_arr[s:e])
                groups_val.append(group_arr[group_idx])
                f_val.append(f_count_arr[group_idx])
                val_group_ids.append(group_idx)

        for formula in test_formulas:
            for group_idx in formula_to_groups[formula]:
                s, e = group_indices[group_idx]
                X_test.extend(X_arr[s:e])
                y_test.extend(y_arr[s:e])
                groups_test.append(group_arr[group_idx])
                f_test.append(f_count_arr[group_idx])
                test_group_ids.append(group_idx)

    return (
        np.asarray(X_train, dtype=np.float32),
        np.asarray(X_val, dtype=np.float32),
        np.asarray(X_test, dtype=np.float32),
        np.asarray(y_train),
        np.asarray(y_val),
        np.asarray(y_test),
        groups_train,
        groups_val,
        groups_test,
        np.asarray(f_train),
        np.asarray(f_val),
        np.asarray(f_test),
        train_formulas_set,
        val_formulas_set,
        test_formulas_set,
    )


def random_oversample_by_f(
    X_train, y_train, groups_train, f_train,
    oversample_ratios, random_state=24,
):
    np.random.seed(random_state)
    group_data = []
    start_idx = 0
    for i, gs in enumerate(groups_train):
        end_idx = start_idx + gs
        group_data.append({
            "X": X_train[start_idx:end_idx],
            "y": y_train[start_idx:end_idx],
            "f": f_train[i],
            "size": gs,
        })
        start_idx = end_idx

    groups_by_f = {}
    for i, gd in enumerate(group_data):
        f_val = gd["f"]
        f_key = 4 if f_val >= 4 else f_val
        groups_by_f.setdefault(f_key, []).append(i)

    oversampled_groups = []
    for f_key in sorted(groups_by_f.keys()):
        group_indices = groups_by_f[f_key]
        ratio = oversample_ratios.get(f_key, 1.0)

        if ratio == 1.0:
            for idx in group_indices:
                oversampled_groups.append(group_data[idx])
        else:
            n_original = len(group_indices)
            n_target = int(n_original * ratio)
            n_oversample = n_target - n_original
            for idx in group_indices:
                oversampled_groups.append(group_data[idx])
            if n_oversample > 0:
                sampled = np.random.choice(group_indices, size=n_oversample, replace=True)
                for idx in sampled:
                    oversampled_groups.append(group_data[idx])

    X_os, y_os, groups_os, f_os = [], [], [], []
    for gd in oversampled_groups:
        X_os.extend(gd["X"])
        y_os.extend(gd["y"])
        groups_os.append(gd["size"])
        f_os.append(gd["f"])

    return np.asarray(X_os), np.asarray(y_os), groups_os, np.asarray(f_os)


def calculate_sample_weights(y_train_os, f_train_os, groups_train_os):
    f_sample_counts = {}
    start_idx = 0
    for i, gs in enumerate(groups_train_os):
        end_idx = start_idx + gs
        f_val = int(f_train_os[i])
        f_sample_counts[f_val] = f_sample_counts.get(f_val, 0) + gs
        start_idx = end_idx

    total_samples = len(y_train_os)
    n_classes = len(f_sample_counts)
    f_weights = {
        f_val: total_samples / (n_classes * cnt)
        for f_val, cnt in f_sample_counts.items()
    }
    sample_weights = []
    for i, gs in enumerate(groups_train_os):
        weight = f_weights[int(f_train_os[i])]
        sample_weights.extend([weight] * gs)
    w = np.asarray(sample_weights, dtype=np.float64)
    return w / w.mean()


def slice_ms1_ms2(X):
    """52-d combined feature (drop col 0 isotope)."""
    return X[:, list(range(1, 53))]


def train_model(
    X_train, X_val, y_train, y_val,
    groups_train, groups_val, f_train, f_val,
    oversample_ratios, random_state,
    n_jobs_lgb: int = 1,
):
    print(f"\nOversampling training groups...")
    X_os, y_os, groups_os, f_os = random_oversample_by_f(
        X_train, y_train, groups_train, f_train,
        oversample_ratios=oversample_ratios,
        random_state=random_state,
    )
    print(f"  After oversample: n_rows={len(y_os)}, n_groups={len(groups_os)}")

    print("Computing sample weights...")
    sw = calculate_sample_weights(y_os, f_os, groups_os)
    print(f"  weight min/max: {sw.min():.4f} / {sw.max():.4f}")

    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        learning_rate=0.01,
        num_leaves=2000,
        max_depth=-1,
        min_data_in_leaf=20,
        max_bin=200,
        bagging_fraction=0.8,
        bagging_freq=1,
        feature_fraction=0.7,
        lambda_l1=0,
        lambda_l2=0,
        random_state=random_state,
        n_estimators=1500,
        n_jobs=n_jobs_lgb,
        verbose=-1,
    )
    t0 = time.time()
    fit_kw = dict(
        X=X_os,
        y=y_os,
        group=groups_os,
        sample_weight=sw,
        eval_set=[(X_val, y_val)],
        eval_group=[groups_val],
        eval_metric="ndcg",
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False)
        ],
    )
    # Sklearn API prefers eval_at over ndcg_eval_at in estimator params (avoids UserWarning).
    try:
        if "eval_at" in inspect.signature(model.fit).parameters:
            fit_kw["eval_at"] = [1]
    except (TypeError, ValueError):
        pass
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Found 'ndcg_eval_at' in params",
            category=UserWarning,
            module=r"lightgbm\.sklearn",
        )
        model.fit(**fit_kw)
    elapsed = time.time() - t0
    bi = getattr(model, "best_iteration_", None) or getattr(model, "n_estimators", None)
    print(f"Training done in {elapsed:.1f}s; best_iteration={bi}")
    return model


def evaluate_top1(y_test, pred_scores, groups_test, f_test):
    group_accs = []
    group_f = []
    start = 0
    for i, gs in enumerate(int(g) for g in groups_test):
        end = start + gs
        true_labels = y_test[start:end]
        scores = pred_scores[start:end]
        ok = bool(true_labels[np.argmax(scores)] == 1)
        group_accs.append(ok)
        group_f.append(int(f_test[i]))
        start = end
    group_accs = np.asarray(group_accs)
    group_f = np.asarray(group_f)
    by_f = {}
    for f_val in np.unique(group_f):
        m = group_f == f_val
        if np.any(m):
            by_f[int(f_val)] = {
                "accuracy": float(np.mean(group_accs[m])),
                "count": int(np.sum(m)),
            }
    return {
        "overall_accuracy": float(np.mean(group_accs)),
        "by_f": by_f,
    }


def main():
    ap = argparse.ArgumentParser(description="Train MS1+MS2 LightGBM ranker.")
    ap.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Folder with X_arr.joblib, y_arr.joblib, group_arr.joblib, formula_strs.joblib",
    )
    ap.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Where to save model (default: <data-dir>/trained_models)",
    )
    ap.add_argument("--random-seed", type=int, default=24)
    ap.add_argument("--val-size", type=float, default=0.1)
    ap.add_argument("--test-size", type=float, default=0.1)
    ap.add_argument(
        "--no-mmap",
        action="store_true",
        help="Load joblib arrays without mmap (try if mmap fails on a mounted filesystem)",
    )
    ap.add_argument(
        "--lgb-jobs",
        type=int,
        default=1,
        help="LightGBM n_jobs (default 1; use -1 only if stable on your OS)",
    )
    args = ap.parse_args()

    faulthandler.enable()

    data_dir = args.data_dir.resolve()
    model_dir = (args.model_dir or (data_dir / "trained_models")).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    use_mmap = not args.no_mmap
    print(f"Loading {data_dir} (joblib mmap={use_mmap}) ...")
    sys.stdout.flush()

    print("  X_arr.joblib ...")
    sys.stdout.flush()
    X_arr = _joblib_load(data_dir / "X_arr.joblib", mmap=use_mmap)
    X_arr = _as_writeable_array(X_arr)

    print("  y_arr.joblib ...")
    sys.stdout.flush()
    y_arr = np.asarray(_joblib_load(data_dir / "y_arr.joblib", mmap=use_mmap))

    print("  group_arr.joblib ...")
    sys.stdout.flush()
    group_arr = np.asarray(_joblib_load(data_dir / "group_arr.joblib", mmap=use_mmap))

    print("  formula_strs.joblib ...")
    sys.stdout.flush()
    formula_strs = _joblib_load(data_dir / "formula_strs.joblib", mmap=False)
    if X_arr.shape[1] < 53:
        print(f"ERROR: X_arr has {X_arr.shape[1]} columns; need >= 53 for this script.")
        return 1
    print(f"X shape={X_arr.shape}, positives={np.sum(y_arr)}")

    true_formulas_per_group = []
    start_idx = 0
    for group_size in group_arr:
        true_formulas_per_group.append(formula_strs[start_idx])
        start_idx += group_size
    f_count_arr = np.array([extract_f_count(f) for f in true_formulas_per_group])

    f_strata = [(0, 0), (1, 1), (2, 2), (3, 3), (4, None)]
    print("Stratified split by formula...")
    splits = stratified_split_by_formula(
        X_arr, y_arr, group_arr, true_formulas_per_group, f_count_arr,
        val_size=args.val_size, test_size=args.test_size,
        f_strata=f_strata,
        random_state=args.random_seed,
    )
    (
        X_train_full,
        X_val_full,
        X_test_full,
        y_train,
        y_val,
        y_test,
        groups_train,
        groups_val,
        groups_test,
        f_train,
        f_val,
        f_test,
        train_f,
        val_f,
        test_f,
    ) = splits

    ov_tv = train_f & val_f
    ov_tt = train_f & test_f
    ov_vt = val_f & test_f
    if ov_tv or ov_tt or ov_vt:
        print("WARN: formula overlap train/val/test:", len(ov_tv), len(ov_tt), len(ov_vt))
    else:
        print("Formula sets disjoint across splits.")

    X_train = slice_ms1_ms2(X_train_full)
    X_val = slice_ms1_ms2(X_val_full)
    X_test = slice_ms1_ms2(X_test_full)

    oversample_ratios = {0: 1.0, 1: 1.0, 2: 1.5, 3: 1.5, 4: 15.0}
    model = train_model(
        X_train,
        X_val,
        y_train,
        y_val,
        groups_train,
        groups_val,
        f_train,
        f_val,
        oversample_ratios,
        args.random_seed,
        n_jobs_lgb=args.lgb_jobs,
    )

    out_model = model_dir / "model_ms1_ms2_ranker.joblib"
    joblib.dump(model, out_model)
    print(f"Saved model -> {out_model}")

    pred = _lgb_predict(model, X_test)
    results = evaluate_top1(y_test, pred, groups_test, f_test)
    print("\ntest Top-1 (by group)")
    print(f"  Overall: {results['overall_accuracy']:.4f}")
    for fv in sorted(results["by_f"]):
        b = results["by_f"][fv]
        print(f"  F={fv}: acc={b['accuracy']:.4f} (n_groups={b['count']})")

    summary_path = model_dir / "train_ms1_ms2_summary.txt"
    lines = [
        f"data_dir={data_dir}",
        f"overall_top1={results['overall_accuracy']:.6f}",
    ]
    for fv in sorted(results["by_f"]):
        b = results["by_f"][fv]
        lines.append(f"F_{fv}_top1={b['accuracy']:.6f}_n={b['count']}")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved summary -> {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
