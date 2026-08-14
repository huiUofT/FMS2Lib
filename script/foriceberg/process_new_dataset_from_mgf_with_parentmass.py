#!/usr/bin/env python3
"""Build ICEBERG spec_datasets from MGF (labels.tsv, spec_files.hdf5, split TSV).

See repo ``README.md``.
"""
import argparse
import os
import re
import sys
from collections import Counter
from pathlib import Path

from foriceberg_paths import data_dir, github_root, pythonpath_src, repo_root

if str(pythonpath_src()) not in sys.path:
    sys.path.insert(0, pythonpath_src())

import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem

import ms_pred.common as common


def parse_mgf_file(mgf_path):
    print(f"Parsing MGF: {mgf_path}")

    spectra = []
    current_spec = {}
    peaks = []

    with open(mgf_path, "r") as f:
        for line in f:
            line = line.strip()

            if line == "BEGIN IONS":
                current_spec = {}
                peaks = []
            elif line.startswith("IDENTIFIER="):
                current_spec["spec"] = line.replace("IDENTIFIER=", "")
            elif line.startswith("SMILES="):
                current_spec["smiles"] = line.replace("SMILES=", "")
            elif line.startswith("PRECURSOR_MZ="):
                current_spec["precursor_mz"] = float(line.replace("PRECURSOR_MZ=", ""))
            elif line.startswith("CHARGE="):
                charge_str = line.replace("CHARGE=", "").replace("+", "")
                current_spec["charge"] = int(charge_str) if charge_str.isdigit() else 1
            elif line.startswith("ADDUCT="):
                current_spec["adduct"] = line.replace("ADDUCT=", "")
            elif line.startswith("INSTRUMENT_TYPE="):
                current_spec["instrument"] = line.replace("INSTRUMENT_TYPE=", "")
            elif line.startswith("FEATURE_ID="):
                current_spec["feature_id"] = line.replace("FEATURE_ID=", "")
            elif line.startswith("COLLISION_ENERGY=") or line.startswith("CE="):
                ce_str = line.replace("COLLISION_ENERGY=", "").replace("CE=", "").strip()
                current_spec["collision_energy"] = ce_str
            elif re.match(r"^\d+\.?\d*\s+\d+\.?\d*", line):
                parts = line.split()
                if len(parts) >= 2:
                    peaks.append((float(parts[0]), float(parts[1])))
            elif line == "END IONS":
                if "spec" in current_spec and "smiles" in current_spec and len(peaks) > 0:
                    current_spec["peaks"] = peaks
                    spectra.append(current_spec.copy())

    print(f"Parsed {len(spectra)} spectra")
    return spectra


def create_labels_tsv(spectra, output_path):
    print(f"\nWriting labels.tsv: {output_path}")

    labels_data = []

    for spec in tqdm(spectra, desc="labels"):
        smiles = spec["smiles"]

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                inchikey = Chem.MolToInchiKey(mol)
            else:
                inchikey = None
        except Exception:
            inchikey = None

        f_count = smiles.count("F")
        collision_energy = spec.get("collision_energy", "30.0")

        instrument = spec.get("instrument", "Orbitrap") or "Orbitrap"
        # Training embeddings only know Orbitrap/QTOF; HCD (Orbitrap fragmentation) → Orbitrap.
        if str(instrument).strip().upper() in {"HCD", "HCD (ORBITRAP)", "ORBITRAP-HCD"}:
            instrument = "Orbitrap"

        labels_data.append(
            {
                "spec": spec["spec"],
                "smiles": smiles,
                "inchikey": inchikey if inchikey else "",
                "ionization": spec.get("adduct", "[M+H]+"),
                "precursor": spec["precursor_mz"],
                "instrument": instrument,
                "collision_energies": f"['{collision_energy}']",
                "f_count": f_count,
            }
        )

    df = pd.DataFrame(labels_data)

    df_labels = df.drop(columns=["f_count"])
    df_labels.to_csv(output_path, sep="\t", index=False)
    print(f"Saved {len(df_labels)} rows to {output_path}")

    f_dist = Counter(df["f_count"].values)
    print("\nFluorine count distribution:")
    for f_count in sorted(f_dist.keys()):
        count = f_dist[f_count]
        percentage = count / len(df) * 100
        print(f"  {f_count} F: {count:6d} ({percentage:5.2f}%)")

    return df


def create_spec_files_hdf5(spectra, output_path):
    print(f"\nWriting spec_files.hdf5: {output_path}")

    h5_dataset = common.HDF5Dataset(output_path, mode="w")

    for spec in tqdm(spectra, desc="spec_files"):
        spec_name = spec["spec"]

        sirius_lines = []

        sirius_lines.append(f"#INSTRUMENT TYPE {spec.get('instrument', 'Orbitrap')}")
        prec_mz_line = f"#PRECURSORMZ {spec['precursor_mz']}"
        sirius_lines.append(prec_mz_line)
        sirius_lines.append(f"#parentmass {prec_mz_line.split(' ', 1)[1]}")
        sirius_lines.append(f"#CHARGE {spec.get('charge', 1)}+")
        sirius_lines.append(f"#ADDUCT {spec.get('adduct', '[M+H]+')}")
        sirius_lines.append(f"#IDENTIFIER {spec_name}")
        sirius_lines.append(f"#SMILES {spec['smiles']}")
        if "feature_id" in spec:
            sirius_lines.append(f"#FEATURE_ID {spec['feature_id']}")
        collision_energy = spec.get("collision_energy", "30.0")
        sirius_lines.append(f"#COLLISION ENERGY {collision_energy}")

        sirius_lines.append("")

        sirius_lines.append(f">collision {collision_energy}")

        for mz, intensity in spec["peaks"]:
            sirius_lines.append(f"{mz} {intensity}")

        sirius_str = "\n".join(sirius_lines)
        h5_dataset.write_str(spec_name, sirius_str)

    h5_dataset.close()
    print(f"Saved {len(spectra)} spectra to {output_path}")


def stratified_split_by_f_count(df, train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=1):
    print("\nStratified split by fluorine count in SMILES")
    print(f"Train: {train_frac*100:.1f}%, val: {val_frac*100:.1f}%, test: {test_frac*100:.1f}%")

    np.random.seed(seed)

    train_inds = []
    val_inds = []
    test_inds = []

    f_groups = df.groupby("f_count")

    print("\nPer-group split:")
    print("-" * 80)

    for f_count, group_df in f_groups:
        n = len(group_df)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        n_test = n - n_train - n_val

        indices = group_df.index.values.copy()
        np.random.shuffle(indices)

        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]

        train_inds.extend(train_idx)
        val_inds.extend(val_idx)
        test_inds.extend(test_idx)

        print(
            f"  {f_count} F: total={n:5d}, train={len(train_idx):5d}, val={len(val_idx):5d}, test={len(test_idx):5d}"
        )

    all_inds = train_inds + val_inds + test_inds
    all_folds = ["train"] * len(train_inds) + ["val"] * len(val_inds) + ["test"] * len(test_inds)

    split_data = {"spec": df.loc[all_inds, "spec"].values, "Fold_0": all_folds}

    split_df = pd.DataFrame(split_data)
    split_df = split_df.sort_values("spec", ascending=True)

    print("\nOverall:")
    print(f"  train: {len(train_inds):6d} ({len(train_inds)/len(df)*100:.2f}%)")
    print(f"  val:   {len(val_inds):6d} ({len(val_inds)/len(df)*100:.2f}%)")
    print(f"  test:  {len(test_inds):6d} ({len(test_inds)/len(df)*100:.2f}%)")

    return split_df, train_inds, val_inds, test_inds


def main():
    parser = argparse.ArgumentParser(
        description="Build spec_datasets from MGF with #parentmass in spec_files (aligned with F_ICEBERG_dataset)."
    )
    parser.add_argument("--mgf-file", required=True, help="Input MGF path")
    parser.add_argument("--dataset-name", default="fluorine_dataset", help="Dataset name under data/spec_datasets/")
    parser.add_argument("--train-frac", type=float, default=0.8, help="Train fraction")
    parser.add_argument("--val-frac", type=float, default=0.1, help="Validation fraction")
    parser.add_argument("--test-frac", type=float, default=0.1, help="Test fraction")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for split")

    args = parser.parse_args()

    root = repo_root()
    os.chdir(root)

    out_data = data_dir() / "spec_datasets" / args.dataset_name
    out_data.mkdir(parents=True, exist_ok=True)
    (out_data / "splits").mkdir(exist_ok=True)

    print("=" * 80)
    print(f"Dataset: {args.dataset_name} (with #parentmass)")
    print("=" * 80)

    spectra = parse_mgf_file(args.mgf_file)

    labels_path = out_data / "labels.tsv"
    df = create_labels_tsv(spectra, labels_path)

    spec_files_path = out_data / "spec_files.hdf5"
    create_spec_files_hdf5(spectra, spec_files_path)

    split_df, train_inds, val_inds, test_inds = stratified_split_by_f_count(
        df, args.train_frac, args.val_frac, args.test_frac, args.seed
    )

    split_path = out_data / "splits" / f"split_{args.seed}.tsv"
    split_df.to_csv(split_path, sep="\t", index=False)
    print(f"\nSaved split file: {split_path}")

    print("\n" + "=" * 80)
    print("Done.")
    print("=" * 80)
    print(f"Output directory: {out_data}")
    print(f"labels.tsv:       {labels_path}")
    print(f"spec_files.hdf5:  {spec_files_path}")
    print(f"split TSV:        {split_path}")
    print("\nTypical next steps:")
    print("  1. Run MAGMA for fragmentation trees")
    print("  2. Train the generation model")
    print("  3. Predict the training set with the generator")
    print("  4. Add intensity labels")
    print("  5. Train the intensity model")


if __name__ == "__main__":
    main()
