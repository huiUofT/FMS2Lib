#!/usr/bin/env python3
"""This code was adapted from the ICEBERG implementation developed by
Goldman et al.

Original repository:
https://github.com/samgoldman97/ms-pred

The ICEBERG model was retrained using an in-house MS/MS dataset of
low-fluorinated compounds. Modifications include dataset preparation,
data splitting, model training, and performance evaluation.

Modifications by: Tao Yang
Date: 20260819

The original copyright notice and license terms are retained.
"""

from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

# FMS2Lib repo root via foriceberg_paths
from foriceberg_paths import github_root, mode_dir, pythonpath_src, repo_root

REPO_ROOT = repo_root()
_DEFAULT_GEN_CKPT = mode_dir() / "generation" / "best.ckpt"
_DEFAULT_INTEN_CKPT = mode_dir() / "intensity" / "best.ckpt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SMILES -> predicted MS2 -> MGF (ICEBERG)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--smiles", type=str, help="Single SMILES string")
    g.add_argument(
        "--smiles-file",
        type=Path,
        help="Text file: one SMILES per line; empty lines and lines starting with # are skipped",
    )
    p.add_argument(
        "--out",
        "--output",
        type=Path,
        required=True,
        dest="out",
        help="Output MGF path",
    )
    p.add_argument(
        "--identifier-prefix",
        type=str,
        default="PRED",
        help="Prefix for IDENTIFIER=/FEATURE_ID= in MGF (single molecule -> PREFIX_0)",
    )
    p.add_argument(
        "--gen-checkpoint",
        type=Path,
        default=_DEFAULT_GEN_CKPT,
        help=f"Default: {_DEFAULT_GEN_CKPT}",
    )
    p.add_argument(
        "--inten-checkpoint",
        type=Path,
        default=_DEFAULT_INTEN_CKPT,
        help=f"Default: {_DEFAULT_INTEN_CKPT}",
    )
    p.add_argument(
        "--collision-energies",
        type=str,
        default="['30.0']",
        help=(
            "Collision energies as a Python literal list of strings, e.g. "
            "['10.0','30.0','50.0']. Default ['30.0'] (30 eV), matching training/eval."
        ),
    )
    p.add_argument("--instrument", type=str, default="Orbitrap")
    p.add_argument("--ionization", type=str, default="[M+H]+")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--sparse-k", type=int, default=100)
    p.add_argument("--max-nodes", type=int, default=2000)
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument(
        "--normalize-max",
        action="store_true",
        help="Scale each spectrum so the maximum intensity is 1",
    )
    return p.parse_args()


def load_smiles_list(args: argparse.Namespace) -> List[str]:
    if args.smiles:
        return [args.smiles.strip()]
    lines = []
    with args.smiles_file.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            lines.append(s)
    if not lines:
        sys.exit("No SMILES read from input")
    return lines


def validate_smiles(s: str) -> str:
    m = Chem.MolFromSmiles(s)
    if m is None:
        sys.exit(f"Invalid SMILES: {s!r}")
    return Chem.MolToSmiles(m)


def precursor_mz_for_smiles(smi: str, adduct: str) -> float:
    sys.path.insert(0, pythonpath_src())
    import ms_pred.common as common
    from ms_pred.common import chem_utils

    mz = chem_utils.mass_from_smi(smi)
    return float(mz + common.ion2mass[adduct])


def build_labels_tsv(
    path: Path,
    smiles_list: Sequence[str],
    prefix: str,
    instrument: str,
    ionization: str,
    ce_str: str,
) -> List[str]:
    """Write labels.tsv; return spec names in row order."""
    spec_names: List[str] = []
    rows = []
    for i, smi in enumerate(smiles_list):
        spec = f"{prefix}_{i}"
        spec_names.append(spec)
        rows.append(
            {
                "spec": spec,
                "smiles": smi,
                "ionization": ionization,
                "instrument": instrument,
                "collision_energies": ce_str,
            }
        )
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    return spec_names


def run_predictor(
    labels_tsv: Path,
    save_dir: Path,
    args: argparse.Namespace,
) -> Path:
    pred_py = Path(pythonpath_src()) / "ms_pred/dag_pred/predict_smis_trained_inten.py"
    if not pred_py.is_file():
        sys.exit(f"Predict script not found: {pred_py}")

    cmd = [
        sys.executable,
        str(pred_py),
        "--dataset-labels",
        str(labels_tsv),
        "--save-dir",
        str(save_dir),
        "--gen-checkpoint",
        str(args.gen_checkpoint),
        "--inten-checkpoint",
        str(args.inten_checkpoint),
        "--sparse-out",
        "--sparse-k",
        str(args.sparse_k),
        "--num-workers",
        str(args.num_workers),
        "--batch-size",
        str(args.batch_size),
        "--max-nodes",
        str(args.max_nodes),
        "--threshold",
        str(args.threshold),
        "--adduct-shift",
    ]
    if args.gpu:
        cmd.insert(2, "--gpu")

    env = os.environ.copy()
    env["PYTHONPATH"] = pythonpath_src() + os.pathsep + env.get("PYTHONPATH", "")

    subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)
    out_h5 = save_dir / "preds.hdf5"
    if not out_h5.is_file():
        sys.exit(f"Predictor did not write {out_h5}")
    return out_h5


def read_spec_mx2_from_h5(
    h5: h5py.File, spec_name: str, ces: List[float]
) -> List[Tuple[np.ndarray, float]]:
    """Return list of (N,2) m/z-intensity arrays and CE, in CE order."""
    out: List[Tuple[np.ndarray, float]] = []
    root_name = f"pred_{spec_name}"
    if root_name not in h5:
        keys_sample = list(h5.keys())[:5]
        sys.exit(f"HDF5 missing group {root_name!r}; top-level keys sample: {keys_sample}")

    g1 = h5[root_name]
    ikey_group_name = None
    for k in g1.keys():
        if k.startswith("ikey "):
            ikey_group_name = k
            break
    if ikey_group_name is None:
        sys.exit(f"No ikey subgroup under {root_name}")

    g2 = g1[ikey_group_name]
    for ce in ces:
        coll_key = f"collision {ce}"
        if coll_key not in g2:
            alt = None
            for ck in g2.keys():
                if ck.startswith("collision"):
                    try:
                        cev = float(ck.replace("collision", "").strip())
                    except ValueError:
                        continue
                    if abs(cev - ce) < 1e-3:
                        alt = ck
                        break
            if alt is None:
                sys.exit(f"No collision energy {ce}; available keys: {list(g2.keys())}")
            coll_key = alt
        ds = g2[coll_key]["spec"]
        data = np.asarray(ds[()])
        if data.ndim != 2 or data.shape[1] < 2:
            sys.exit(f"unexpected spec shape: {data.shape}")
        out.append((data[:, :2].astype(np.float64), float(ce)))
    return out


def write_mgf_block(
    fp,
    identifier: str,
    smiles: str,
    precursor_mz: float,
    peak_mx: np.ndarray,
    collision_energy: float,
    normalize_max: bool,
) -> None:
    if peak_mx.size == 0:
        mz_i = np.zeros((0, 2))
    else:
        mz_i = peak_mx.copy()
        if normalize_max and mz_i.shape[0] > 0:
            m = float(np.max(mz_i[:, 1]))
            if m > 0:
                mz_i[:, 1] = mz_i[:, 1] / m

    fp.write("BEGIN IONS\n")
    fp.write("CHARGE=1+\n")
    fp.write(f"PRECURSOR_MZ={precursor_mz:.7f}\n")
    fp.write(f"IDENTIFIER={identifier}\n")
    fp.write("ADDUCT=[M+H]+\n")
    fp.write("INSTRUMENT_TYPE=Orbitrap\n")
    fp.write(f"FEATURE_ID={identifier}\n")
    fp.write(f"SMILES={smiles}\n")
    fp.write("IONMODE=positive\n")
    fp.write(f"COLLISION_ENERGY={collision_energy}\n")
    for row in mz_i:
        fp.write(f"{float(row[0]):.10g} {float(row[1]):.10g}\n")
    fp.write("END IONS\n")


def main() -> None:
    args = parse_args()
    for label, p in (("GEN", args.gen_checkpoint), ("INTEN", args.inten_checkpoint)):
        if not p.is_file():
            sys.exit(f"Missing {label} checkpoint: {p}")
    smis = [validate_smiles(s) for s in load_smiles_list(args)]

    try:
        ces = [float(x) for x in ast.literal_eval(args.collision_energies)]
    except (ValueError, SyntaxError) as e:
        sys.exit(f"--collision-energies must be a Python literal list; parse error: {e}")

    sys.path.insert(0, pythonpath_src())

    with tempfile.TemporaryDirectory(prefix="mgf_pred_") as tmpd:
        tmpdir = Path(tmpd)
        labels_tsv = tmpdir / "labels.tsv"
        spec_names = build_labels_tsv(
            labels_tsv,
            smis,
            args.identifier_prefix,
            args.instrument,
            args.ionization,
            args.collision_energies,
        )
        h5_path = run_predictor(labels_tsv, tmpdir, args)
        with h5py.File(h5_path, "r") as h5, args.out.open("w", encoding="utf-8") as out:
            for spec_name, smi in zip(spec_names, smis):
                pmz = precursor_mz_for_smiles(smi, args.ionization)
                blocks = read_spec_mx2_from_h5(h5, spec_name, ces)
                for arr, ce in blocks:
                    block_id = (
                        f"{spec_name}_ce{ce:g}"
                        if len(ces) > 1
                        else spec_name
                    )
                    write_mgf_block(
                        out,
                        identifier=block_id,
                        smiles=smi,
                        precursor_mz=pmz,
                        peak_mx=arr,
                        collision_energy=ce,
                        normalize_max=args.normalize_max,
                    )

    print(f"Wrote: {args.out.resolve()}")


if __name__ == "__main__":
    main()
