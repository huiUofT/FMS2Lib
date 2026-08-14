#!/usr/bin/env python3
"""Assign subformulae from labels + spec_files.hdf5 (docx training step).

Writes ``{data-dir}/subformulae/{output-dir-name}`` (default ``no_subform.hdf5``).
"""
from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from tqdm import tqdm

from foriceberg_paths import pythonpath_src

_SRC = pythonpath_src()
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from ms_pred import common  # noqa: E402
from ms_pred.magma import fragmentation  # noqa: E402


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assign subformulae / no_subform tables for ICEBERG intensity labeling."
    )
    parser.add_argument("--data-dir", required=True, help="Dataset dir containing spec_files.hdf5")
    parser.add_argument(
        "--labels-file",
        required=True,
        help="labels.tsv with spec, smiles, ionization columns",
    )
    parser.add_argument(
        "--output-dir-name",
        default="no_subform.hdf5",
        help="File name under data-dir/subformulae/ (default: no_subform.hdf5)",
    )
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument(
        "--use-all",
        action="store_true",
        default=False,
        help="Keep peaks without formula assignment (raw-like table).",
    )
    parser.add_argument("--use-magma", action="store_true", default=False)
    parser.add_argument("--mass-diff-type", default="ppm", type=str)
    parser.add_argument("--mass-diff-thresh", default=10, type=float)
    parser.add_argument("--inten-thresh", default=0.001, type=float)
    parser.add_argument("--max-formulae", default=50, type=int)
    parser.add_argument("--num-workers", default=8, type=int)
    return parser.parse_args()


def get_output_dict(
    spec_name: str,
    spec: np.ndarray,
    form: str,
    mass_diff_type: str,
    mass_diff_thresh: float,
    inten_thresh: float,
    adduct_type: str,
    max_formulae: int = 100,
    use_all=False,
    smiles: str = "",
    use_magma: bool = False,
) -> dict:
    empty_output_dict = {
        "cand_form": form,
        "spec_name": spec_name,
        "cand_ion": adduct_type,
        "output_tbl": None,
    }
    if adduct_type not in common.ion2mass:
        return empty_output_dict
    if spec is None:
        return empty_output_dict

    spec = common.max_inten_spec(spec, max_formulae, inten_thresh=inten_thresh)
    spec_masses, spec_intens = spec[:, 0], spec[:, 1]
    adduct_masses = common.ion2mass[adduct_type]
    e_adduct_masses = (
        -common.ELECTRON_MASS if common.is_positive_adduct(adduct_type) else common.ELECTRON_MASS
    )

    if use_all:
        output_tbl = {
            "mz": list(spec_masses),
            "ms2_inten": list(spec_intens),
            "rel_inten": list(spec_intens),
            "mono_mass": list(spec_masses),
            "formula_mass_no_adduct": list(spec_masses - adduct_masses),
            "mass_diff": [0] * len(spec_masses),
            "formula": [""] * len(spec_masses),
            "ions": [adduct_masses] * len(spec_masses),
        }
        if len(spec_intens) == 0:
            output_tbl = None
        return {
            "cand_form": form,
            "spec_name": spec_name,
            "cand_ion": adduct_type,
            "output_tbl": output_tbl,
        }

    if use_magma:
        fe = fragmentation.FragmentEngine(mol_str=smiles)
        try:
            fe.generate_fragments()
        except Exception:
            print(f"Error with generating fragments for spec {smiles}")
            return {
                "cand_form": form,
                "spec_name": spec_name,
                "cand_ion": adduct_type,
                "output_tbl": None,
            }
        cross_prod, masses = fe.get_frag_forms()
    else:
        cross_prod, masses = common.get_all_subsets(form)

    masses_with_adduct = masses + adduct_masses
    masses_with_e_adduct = masses + e_adduct_masses
    adduct_types = np.array([adduct_type] * len(masses_with_adduct))
    mass_diffs = np.minimum(
        np.abs(spec_masses[:, None] - masses_with_adduct[None, :]),
        np.abs(spec_masses[:, None] - masses_with_e_adduct[None, :]),
    )

    formula_inds = mass_diffs.argmin(-1)
    min_mass_diff = mass_diffs[np.arange(len(mass_diffs)), formula_inds]

    if mass_diff_type == "ppm":
        mass_divisior = np.copy(spec_masses)
        mass_divisior[mass_divisior <= 200] = 200
        min_mass_diff = (min_mass_diff / mass_divisior) * 1e6

    valid_mask = min_mass_diff < mass_diff_thresh
    spec_masses = spec_masses[valid_mask]
    spec_intens = spec_intens[valid_mask]
    min_mass_diff = min_mass_diff[valid_mask]
    formula_inds = formula_inds[valid_mask]

    formulas = np.array([common.vec_to_formula(j) for j in cross_prod[formula_inds]])
    formula_masses = masses_with_adduct[formula_inds]
    formula_mass_no_adduct = masses[formula_inds]
    adduct_types = adduct_types[formula_inds]

    formula_idx_dict = {}
    uniq_mask = []
    for idx in range(len(formulas)):
        formula = formulas[idx]
        if formula not in formula_idx_dict:
            uniq_mask.append(True)
            formula_idx_dict[formula] = idx
        else:
            merge_idx = formula_idx_dict[formula]
            uniq_mask.append(False)
            spec_intens[merge_idx] = max(spec_intens[idx], spec_intens[merge_idx])

    spec_masses = spec_masses[uniq_mask]
    spec_intens = spec_intens[uniq_mask]
    min_mass_diff = min_mass_diff[uniq_mask]
    formula_masses = formula_masses[uniq_mask]
    formula_mass_no_adduct = formula_mass_no_adduct[uniq_mask]
    formulas = formulas[uniq_mask]
    adduct_types = adduct_types[uniq_mask]

    if spec_intens.size == 0:
        return {
            "cand_form": form,
            "spec_name": spec_name,
            "cand_ion": adduct_type,
            "output_tbl": None,
        }

    output_tbl = {
        "mz": list(spec_masses),
        "ms2_inten": list(spec_intens),
        "rel_inten": list(spec_intens),
        "mono_mass": list(formula_masses),
        "formula_mass_no_adduct": list(formula_mass_no_adduct),
        "mass_diff": list(min_mass_diff),
        "formula": list(formulas),
        "ions": [adduct for adduct in list(adduct_types)],
    }
    return {
        "cand_form": form,
        "spec_name": spec_name,
        "cand_ion": adduct_type,
        "output_tbl": output_tbl,
    }


def process_spec_file(spec_name: str, data_dir: Path):
    ms_h5 = common.HDF5Dataset(data_dir / "spec_files.hdf5")
    try:
        spec_lines = ms_h5.read_str(f"{spec_name}.ms").split("\n")
    except KeyError:
        spec_lines = ms_h5.read_str(spec_name).split("\n")
    meta, tuples = common.parse_spectra(spec_lines)
    specs = common.process_spec_file(meta, tuples, merge_specs=False)
    return spec_name, specs


def main() -> None:
    args = get_args()
    data_dir = Path(args.data_dir)
    label_path = Path(args.labels_file)
    if not (data_dir / "spec_files.hdf5").is_file():
        sys.exit(f"Missing {data_dir / 'spec_files.hdf5'}")
    if not label_path.is_file():
        sys.exit(f"Missing labels: {label_path}")

    labels_df = pd.read_csv(label_path, sep="\t")
    if args.debug:
        labels_df = labels_df[:5000]

    subform_dir = data_dir / "subformulae"
    subform_dir.mkdir(exist_ok=True)
    output_dir = subform_dir / args.output_dir_name

    spec_fn_lst = labels_df["spec"].to_list()
    print("Processing ms raw data")
    proc_spec_full = partial(process_spec_file, data_dir=data_dir)
    if args.debug or args.num_workers <= 0:
        input_specs = [proc_spec_full(i) for i in tqdm(spec_fn_lst)]
    else:
        input_specs = common.chunked_parallel(
            spec_fn_lst, proc_spec_full, chunks=1000, max_cpu=args.num_workers
        )
    input_specs_dict = {tup[0]: tup[1] for tup in input_specs}

    export_dicts = []
    for _, row in labels_df.iterrows():
        spec_name = row["spec"]
        if "formula" in row and pd.notna(row["formula"]):
            formula = row["formula"]
        else:
            smiles = row["smiles"]
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    print(f"Warning: bad SMILES for {spec_name}, skip")
                    continue
                formula = CalcMolFormula(mol)
            except Exception as exc:
                print(f"Warning: formula failed for {spec_name}: {exc}, skip")
                continue

        if spec_name not in input_specs_dict:
            print(f"Warning: missing spectrum for {spec_name}, skip")
            continue

        for colli_eng, spec in input_specs_dict[spec_name].items():
            export_dicts.append(
                {
                    "spec_name": f"{spec_name}_{colli_eng}",
                    "spec": spec,
                    "form": formula,
                    "mass_diff_type": args.mass_diff_type,
                    "mass_diff_thresh": args.mass_diff_thresh,
                    "inten_thresh": args.inten_thresh,
                    "max_formulae": args.max_formulae,
                    "adduct_type": row["ionization"],
                    "use_all": args.use_all,
                    "smiles": row["smiles"],
                    "use_magma": args.use_magma,
                }
            )

    print(f"There are {len(export_dicts)} spec-cand pairs")
    export_wrapper = lambda x: get_output_dict(**x)
    if args.debug or args.num_workers <= 0:
        output_dict_lst = [export_wrapper(i) for i in export_dicts]
    else:
        output_dict_lst = common.chunked_parallel(
            export_dicts, export_wrapper, chunks=1000, max_cpu=args.num_workers
        )
    assert len(export_dicts) == len(output_dict_lst)

    h5 = common.HDF5Dataset(output_dir, mode="w")
    for output_dict in tqdm(output_dict_lst):
        h5.write_str(f'{output_dict["spec_name"]}.json', json.dumps(output_dict, indent=4))
    h5.close()
    print(f"Wrote: {output_dir}")


if __name__ == "__main__":
    main()
