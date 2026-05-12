#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end eval helper:

1) Call ``generate_eval_joblibs.prepare_training_data`` (**MGF + optional formulas**, or ``--mgf-only``).
2) Run trained *.joblib models (inline ``predict_scores``: no msbuddy/scipy at import).

Supervised (--formulas + optional ``--no-insert-ground-truth``): Top‑1/3/5 vs labels.

Inference (``--mgf-only``): no formula text file; MsBuddy proposes candidates → ranker scores them →
writes ``predictions_<model>.csv`` (identifier, m/z, top deduped formula, score).

Pair with ``train/train_ranker.py``: LambdaRank uses cols 1..52 of ``X_arr`` when width ≥53.

Ranker hooks are defined in this file (avoids importing ``run_outside_eval_expanded_features`` / msbuddy
for ``--skip-generate`` predict). Feature generation still uses ``generate_eval_joblibs`` (lazy when
not ``--skip-generate``).

Standalone MGF caches: ``run_outside_eval_default_features.py``, ``run_outside_eval_expanded_features*.py``.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

_EVAL_DIR = Path(__file__).resolve().parent
for _anc in _EVAL_DIR.parents:
    if (_anc / "src" / "msbuddy").is_dir():
        _s = str(_anc / "src")
        if _s not in sys.path:
            sys.path.insert(0, _s)
        break
    if (_anc / "msbuddy" / "__init__.py").is_file():
        _s = str(_anc)
        if _s not in sys.path:
            sys.path.insert(0, _s)
        break
_ed = str(_EVAL_DIR.resolve())
if _ed not in sys.path:
    sys.path.insert(0, _ed)


def _call_estimator_predict_like(m: Any, name: str, X: np.ndarray) -> Any:
    fn = getattr(m, name)
    kwargs = {}
    try:
        if "validate_features" in inspect.signature(fn).parameters:
            kwargs["validate_features"] = False
    except (TypeError, ValueError):
        pass
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*does not have valid feature names.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module=r"sklearn\.utils\.validation",
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*Found 'ndcg_eval_at' in params.*",
            category=UserWarning,
            module=r"lightgbm\.sklearn",
        )
        return fn(X, **kwargs)


def _unpack_model(obj: Any) -> Any:
    if isinstance(obj, (list, tuple)) and len(obj) == 12:
        return obj[1]
    return obj


def model_expected_n_features(model: Any) -> Optional[int]:
    m = _unpack_model(model)
    if hasattr(m, "n_features_in_"):
        try:
            return int(m.n_features_in_)
        except Exception:
            return None
    try:
        import lightgbm as lgb  # type: ignore

        if isinstance(m, lgb.Booster):
            return int(m.num_feature())
    except Exception:
        pass
    try:
        import xgboost as xgb  # type: ignore

        if isinstance(m, xgb.Booster):
            return int(m.num_features())
    except Exception:
        pass
    return None


def predict_scores(model: Any, X: np.ndarray) -> np.ndarray:
    m = _unpack_model(model)
    if X.size == 0:
        return np.array([])
    if hasattr(m, "predict_proba"):
        p = _call_estimator_predict_like(m, "predict_proba", X)
        if p.ndim == 2 and p.shape[1] >= 2:
            return np.asarray(p[:, 1], dtype=np.float64)
        return np.asarray(p.ravel(), dtype=np.float64)
    try:
        import lightgbm as lgb  # type: ignore

        if isinstance(m, lgb.Booster):
            return np.asarray(m.predict(X), dtype=np.float64)
    except Exception:
        pass
    try:
        import xgboost as xgb  # type: ignore

        if isinstance(m, xgb.Booster):
            return np.asarray(m.predict(xgb.DMatrix(X)), dtype=np.float64)
    except Exception:
        pass
    if hasattr(m, "decision_function"):
        return np.asarray(
            _call_estimator_predict_like(m, "decision_function", X),
            dtype=np.float64,
        )
    return np.asarray(
        _call_estimator_predict_like(m, "predict", X),
        dtype=np.float64,
    )


def accuracy_at_k(pred_formulas: List[str], true_f: str, k: int) -> bool:
    if not true_f:
        return False
    top = pred_formulas[:k]
    return true_f in top


def load_run_meta(data_dir: Path) -> Dict[str, Any]:
    p = data_dir / "run_meta.joblib"
    if not p.is_file():
        return {}
    try:
        return dict(joblib.load(p))
    except Exception:
        return {}


def extract_f_count(formula_str: str) -> int:
    m = re.search(r"F(\d+)", formula_str)
    if m:
        return int(m.group(1))
    if "F" in formula_str:
        return 1
    return 0


def slice_X_for_flat_joblibs(X_block: np.ndarray, n_feat: int) -> Optional[np.ndarray]:
    """Match ``train_ranker.slice_ms1_ms2`` layout (drop col 0 when X has MS1 iso)."""
    if X_block.ndim != 2 or X_block.shape[1] == 0:
        return None
    nc = X_block.shape[1]
    if n_feat == 52 and nc >= 53:
        return np.ascontiguousarray(X_block[:, 1:53], dtype=np.float64)
    if n_feat == 52 and nc == 52:
        return np.ascontiguousarray(X_block, dtype=np.float64)
    if n_feat == 53 and nc >= 53:
        return np.ascontiguousarray(X_block[:, :53], dtype=np.float64)
    if n_feat == int(nc):
        return np.ascontiguousarray(X_block, dtype=np.float64)
    return None


def _joblib_load(path: Path, mmap: bool):
    try:
        if mmap:
            return joblib.load(path, mmap_mode="r")
    except TypeError:
        pass
    return joblib.load(path)


def ranked_formulas_dedupe(formulas: List[str], scores: np.ndarray) -> List[str]:
    order = np.argsort(-scores)
    seen = set()
    ranked: List[str] = []
    for j in order:
        fs = formulas[j]
        if fs not in seen:
            seen.add(fs)
            ranked.append(fs)
    return ranked


def true_formula_from_labels(
    y_slice: np.ndarray,
    formulas_slice: List[str],
    gt_formula: Optional[str],
) -> str:
    if gt_formula:
        return gt_formula.strip()
    pos = np.flatnonzero(y_slice == 1)
    if pos.size > 0:
        return formulas_slice[int(pos[0])]
    return ""


def evaluate_model_on_ranker_joblibs(
    data_dir: Path,
    model_path: Path,
    no_mmap: bool,
) -> Dict[str, Any]:
    mmap = not no_mmap
    raw_model = joblib.load(model_path)
    n_feat = model_expected_n_features(raw_model)
    if n_feat is None:
        return {
            "model": model_path.name,
            "model_path": str(model_path),
            "error": "could_not_infer_n_features",
            "top1": None,
            "top3": None,
            "top5": None,
            "n_evaluated": 0,
        }

    X_arr = np.asarray(_joblib_load(data_dir / "X_arr.joblib", mmap=mmap))
    y_arr = np.asarray(_joblib_load(data_dir / "y_arr.joblib", mmap=mmap))
    group_arr = np.asarray(_joblib_load(data_dir / "group_arr.joblib", mmap=mmap))
    formula_strs: List[str] = list(
        _joblib_load(data_dir / "formula_strs.joblib", mmap=False)
    )
    gt_path = data_dir / "gt_formula_per_group.joblib"
    gt_formulas: Optional[List[str]] = None
    if gt_path.is_file():
        gt_formulas = list(joblib.load(gt_path))

    n_groups = len(group_arr)
    if gt_formulas is not None and len(gt_formulas) != n_groups:
        gt_formulas = None

    top1 = top3 = top5 = 0
    n_eval = 0
    skipped = 0
    by_f_hit: Dict[int, List[float]] = {}
    start = 0
    for gi, g in enumerate(int(x) for x in group_arr):
        end = start + g
        y_sl = y_arr[start:end]
        forms = formula_strs[start:end]

        gt_f = gt_formulas[gi] if gt_formulas is not None else None
        true_f = true_formula_from_labels(y_sl, forms, gt_f)
        if not true_f:
            skipped += 1
            start = end
            continue

        X_blk = slice_X_for_flat_joblibs(X_arr[start:end, :], n_feat)
        if X_blk is None:
            skipped += 1
            start = end
            continue

        scores = predict_scores(raw_model, X_blk)
        if scores.size == 0:
            skipped += 1
            start = end
            continue

        ranked = ranked_formulas_dedupe(forms, scores)
        if ranked[0] == true_f:
            top1 += 1
        if accuracy_at_k(ranked, true_f, 3):
            top3 += 1
        if accuracy_at_k(ranked, true_f, 5):
            top5 += 1

        f_key = extract_f_count(true_f)
        by_f_hit.setdefault(f_key, []).append(1.0 if ranked[0] == true_f else 0.0)
        n_eval += 1
        start = end

    by_f_summary = {
        fk: {"n": len(v), "top1": float(np.mean(v))}
        for fk, v in sorted(by_f_hit.items())
    }

    return {
        "model": model_path.name,
        "model_path": str(model_path),
        "n_features": int(n_feat),
        "n_evaluated": n_eval,
        "skipped_groups": skipped,
        "top1": top1 / n_eval if n_eval else 0.0,
        "top3": top3 / n_eval if n_eval else 0.0,
        "top5": top5 / n_eval if n_eval else 0.0,
        "by_F_top1": by_f_summary,
    }


def write_ranker_predictions_csv(
    data_dir: Path,
    model_path: Path,
    out_csv: Path,
    *,
    no_mmap: bool,
) -> Tuple[bool, str]:
    """
    Dedupe-by-formula ranking per spectrum; export top‑1 predicted formula + score per row.
    """
    mmap = not no_mmap
    raw_model = joblib.load(model_path)
    n_feat = model_expected_n_features(raw_model)
    if n_feat is None:
        return False, "could_not_infer_n_features"

    X_arr = np.asarray(_joblib_load(data_dir / "X_arr.joblib", mmap=mmap))
    group_arr = np.asarray(_joblib_load(data_dir / "group_arr.joblib", mmap=mmap))
    formula_strs: List[str] = list(
        _joblib_load(data_dir / "formula_strs.joblib", mmap=False)
    )

    sid_path = data_dir / "spectrum_ids_per_group.joblib"
    pmz_path = data_dir / "precursor_mz_per_group.joblib"
    spectrum_ids = (
        list(joblib.load(sid_path)) if sid_path.is_file() else [f"group_{i}" for i in range(len(group_arr))]
    )
    precursor_mzs = (
        list(joblib.load(pmz_path)) if pmz_path.is_file()
        else [float("nan")] * len(group_arr)
    )

    rows_out: List[Dict[str, Any]] = []
    start = 0
    for gi, g in enumerate(int(x) for x in group_arr):
        end = start + g
        forms = formula_strs[start:end]
        sid = spectrum_ids[gi] if gi < len(spectrum_ids) else f"group_{gi}"
        pmz = precursor_mzs[gi] if gi < len(precursor_mzs) else float("nan")

        X_blk = slice_X_for_flat_joblibs(X_arr[start:end, :], int(n_feat))
        if X_blk is None:
            rows_out.append(
                {
                    "identifier": sid,
                    "precursor_mz": pmz,
                    "pred_formula": "",
                    "score": "",
                    "error": "feature_shape_mismatch",
                }
            )
            start = end
            continue

        scores = predict_scores(raw_model, X_blk)
        if scores.size == 0:
            rows_out.append(
                {
                    "identifier": sid,
                    "precursor_mz": pmz,
                    "pred_formula": "",
                    "score": "",
                    "error": "empty_scores",
                }
            )
            start = end
            continue

        order = np.argsort(-scores)
        seen_rank: set = set()
        top_f = ""
        top_sc = float("nan")
        for j in order:
            fs = forms[j]
            if fs not in seen_rank:
                seen_rank.add(fs)
                top_f = fs
                top_sc = float(scores[j])
                break

        rows_out.append(
            {
                "identifier": sid,
                "precursor_mz": pmz,
                "pred_formula": top_f,
                "score": top_sc if top_f else "",
                "error": "",
            }
        )
        start = end

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["identifier", "precursor_mz", "pred_formula", "score", "error"],
        )
        w.writeheader()
        w.writerows(rows_out)

    return True, str(out_csv)


def parse_args(argv: Optional[List[str]] = None):
    ap = argparse.ArgumentParser(
        description=(
            "MsBuddy eval: build joblibs (generate_eval_joblibs) + multi-model prediction "
            "(same predict_scores path as run_outside_eval_expanded_features)."
        )
    )
    ap.add_argument(
        "--output-dir",
        "--data-dir",
        type=str,
        dest="output_dir",
        required=True,
        help="Directory with/writing X_arr,y_arr,... joblibs",
    )
    ap.add_argument("--skip-generate", action="store_true", help="Only run prediction step")
    ap.add_argument("--skip-predict", action="store_true", help="Only run feature generation")

    ap.add_argument("--mgf", type=str, default=None, help="MGF (required unless --skip-generate)")
    ap.add_argument(
        "--formulas",
        type=str,
        default=None,
        help="Supervised formulas TXT (omit with --mgf-only)",
    )
    ap.add_argument(
        "--mgf-only",
        action="store_true",
        help="Generate without formula file for ranker inference (writes run_meta.joblib)",
    )
    ap.add_argument(
        "--no-insert-ground-truth",
        action="store_true",
        help="Supervised only: label by matching formulas line to MsBuddy candidates",
    )

    ap.add_argument("--ms1-tol", type=float, default=2.0)
    ap.add_argument("--ms2-tol", type=float, default=5.0)
    ap.add_argument("--random-seed", type=int, default=24)
    ap.add_argument("--n-cpu", type=int, default=-1)
    ap.add_argument("--batch-size", type=int, default=20000)
    ap.add_argument("--max-candidates-per-spectrum", type=int, default=None)

    ap.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help=(
            "Folder scanned with --model-glob for *.joblib (ignored if --model is set)"
        ),
    )
    ap.add_argument(
        "--model",
        dest="model_paths",
        action="append",
        metavar="PATH",
        default=None,
        help=(
            "Explicit path to one ranker .joblib; repeat for multiple models. "
            "If set, skips directory scan (no need for --models-dir)."
        ),
    )
    ap.add_argument("--model-glob", type=str, default="*.joblib")
    ap.add_argument(
        "--no-mmap",
        action="store_true",
        help="Load feature joblibs without mmap",
    )
    ap.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Write prediction summary JSON to this path",
    )
    args = ap.parse_args(argv)
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.output_dir).expanduser().resolve()

    if args.skip_generate and args.skip_predict:
        print("ERROR: specify at most one of --skip-generate / --skip-predict", file=sys.stderr)
        return 2

    if not args.skip_generate:
        from generate_eval_joblibs import prepare_training_data

        if not args.mgf:
            print("ERROR: --mgf required for generate step", file=sys.stderr)
            return 2
        mgf_only = args.mgf_only
        if not mgf_only and not args.formulas:
            print(
                "ERROR: --formulas required unless --mgf-only.",
                file=sys.stderr,
            )
            return 2
        fp = args.formulas if not mgf_only else None
        insert_gt = (not mgf_only) and (not args.no_insert_ground_truth)
        ok = prepare_training_data(
            mgf_path=args.mgf,
            formulas_path=fp,
            output_dir=str(out_dir),
            ms1_tol=args.ms1_tol,
            ms2_tol=args.ms2_tol,
            random_seed=args.random_seed,
            n_cpu=args.n_cpu,
            batch_size=args.batch_size,
            max_candidates_per_spectrum=args.max_candidates_per_spectrum,
            insert_ground_truth=insert_gt,
            mgf_only=mgf_only,
        )
        if not ok:
            return 1

    if args.skip_predict:
        print(f"Skipping predict; artifacts under {out_dir}")
        return 0

    explicit = list(args.model_paths) if args.model_paths else []
    if explicit:
        model_files = [Path(p).expanduser().resolve() for p in explicit]
        for mp in model_files:
            if not mp.is_file():
                print(f"ERROR: model file not found: {mp}", file=sys.stderr)
                return 1
        models_dir_repr = str(model_files[0].parent)
    else:
        if not args.models_dir:
            print(
                "ERROR: prediction needs --model (file path) and/or --models-dir",
                file=sys.stderr,
            )
            return 2
        models_dir = Path(args.models_dir).expanduser().resolve()
        models_dir_repr = str(models_dir)
        model_files = sorted(models_dir.glob(args.model_glob))
        if not model_files:
            print(
                f"No models matching {args.model_glob!r} in {models_dir}",
                file=sys.stderr,
            )
            return 1

    for need in ("X_arr.joblib", "y_arr.joblib", "group_arr.joblib", "formula_strs.joblib"):
        p = out_dir / need
        if not p.is_file():
            print(f"ERROR: missing {p}", file=sys.stderr)
            return 1

    meta = load_run_meta(out_dir)
    inference_mode = bool(meta.get("mgf_only"))

    rows = []
    if inference_mode:
        print("Inference mode (--mgf-only): writing predictions_*.csv (no labelled Top-k).")
        for mp in model_files:
            out_csv = out_dir / f"predictions_{mp.stem}.csv"
            ok, msg = write_ranker_predictions_csv(
                out_dir, mp, out_csv, no_mmap=args.no_mmap
            )
            row = {"model": mp.name, "model_path": str(mp), "prediction_csv": msg if ok else None}
            if not ok:
                row["error"] = msg
                print(f"{mp.name}: {msg}")
                rows.append(row)
                continue
            print(f"{mp.name}: wrote {msg}")
            rows.append(row)
    else:
        for mp in model_files:
            row = evaluate_model_on_ranker_joblibs(
                data_dir=out_dir,
                model_path=mp,
                no_mmap=args.no_mmap,
            )
            rows.append(row)
            er = row.get("error")
            if er:
                print(f"{row['model']}: {er}")
                continue
            print(
                f"{row['model']:40s}  n_feat={row['n_features']}  "
                f"n_eval={row['n_evaluated']}  "
                f"Top1={row['top1']:.4f}  Top3={row['top3']:.4f}  Top5={row['top5']:.4f}  "
                f"skipped={row['skipped_groups']}"
            )

    manifest = {
        "data_dir": str(out_dir),
        "models_dir": models_dir_repr,
        "model_paths": [str(p) for p in model_files],
        "mgf_only": inference_mode,
        "results": rows,
    }
    if args.json_out:
        jo = Path(args.json_out).expanduser().resolve()
        jo.parent.mkdir(parents=True, exist_ok=True)
        jo.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Wrote {jo}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
