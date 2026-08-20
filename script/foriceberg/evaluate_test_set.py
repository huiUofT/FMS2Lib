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

Evaluate test-set predictions with official binned cosine similarity (dist_bin).

Used by ``run_predict_and_evaluate_testset.py`` or standalone after
``preds.hdf5`` exists. See repo README (ICEBERG / script reference).
"""
import sys
from pathlib import Path
import argparse
import json
import numpy as np
from numpy.linalg import norm
import pandas as pd
import h5py
from functools import partial
from typing import Dict, List

from foriceberg_paths import data_spec_datasets, github_root, pythonpath_src, repo_root

project_root = repo_root()
sys.path.insert(0, pythonpath_src())

try:
    from tqdm import tqdm
    USE_TQDM = True
except ImportError:
    print("Warning: tqdm not installed; using simple progress output")
    USE_TQDM = False
    def tqdm(iterable, desc=None, total=None):
        """Minimal tqdm fallback."""
        if desc:
            print(desc)
        if total:
            print(f"Total: {total} items")
        count = 0
        for item in iterable:
            count += 1
            if count % 100 == 0:
                print(f"  processed: {count}/{total if total else '?'}")
            yield item
        if desc:
            print(f"Done: {count} items")

import ms_pred.common as common
from ms_pred.common.splitter import get_splits


def dist_bin(cand_preds_dict: List[Dict], true_spec_dict: dict, sparse=True, ignore_peak=None, func='cos', selected_evs=None, agg=True) -> np.ndarray:
    """
    Binned cosine distance (from retrieval_benchmark_torchmetrics dist_bin).

    Args:
        cand_preds_dict: list of {collision_energy: spectrum} dicts per candidate
        true_spec_dict: ground-truth {collision_energy: binned spectrum}
        sparse: whether predictions are sparse [m/z, intensity] pairs
        ignore_peak: bin index to zero (e.g. precursor)
        func: 'cos' or 'entropy'
        selected_evs: collision energies to include
        agg: aggregate across energies with peak-count weights
        
    Returns:
        np.ndarray: cosine distances for each candidate
    """
    dist = []
    true_npeaks = []

    if selected_evs:
        true_spec_dict = {k: v for k, v in true_spec_dict.items() if str(k) in selected_evs}
    
    true_spec_dict = {f'{float(k):.0f}': v for k, v in true_spec_dict.items()}
    cand_preds_dict = [{f'{float(k):.0f}': v for k, v in cand_dict.items()} for cand_dict in cand_preds_dict]
    
    for colli_eng in true_spec_dict.keys():
        true_spec = true_spec_dict[colli_eng]
        try:
            cand_preds = np.stack([i.get(colli_eng, np.zeros_like(true_spec)) for i in cand_preds_dict], axis=0)
            stacked = True
        except ValueError:
            cand_preds = [i.get(colli_eng, np.zeros((0,2))) for i in cand_preds_dict]
            stacked = False
        
        if sparse:
            pred_specs = np.zeros((len(cand_preds), true_spec.shape[0]))
            if stacked:
                inds = cand_preds[:, :, 0].astype(int)
                pos_1 = np.repeat(np.arange(inds.shape[0])[:, np.newaxis], inds.shape[1], axis=1)
                valid_mask = inds < pred_specs.shape[1]
                pred_specs[pos_1[valid_mask], inds[valid_mask]] = cand_preds[:, :, 1][valid_mask]
            else:
                for i, cand_pred in enumerate(cand_preds):
                    if cand_pred.ndim == 2 and cand_pred.shape[1] == 2:
                        inds = cand_pred[:, 0].astype(int)
                        valid_mask = inds < pred_specs.shape[1]
                        pred_specs[i, inds[valid_mask]] = cand_pred[valid_mask, 1]
        else:
            pred_specs = cand_preds

        if ignore_peak:
            pred_specs[:, int(ignore_peak):] = 0
            true_spec = true_spec.copy()
            true_spec[int(ignore_peak):] = 0

        true_npeaks.append(np.sum(true_spec > 0))

        if func == 'cos':
            # Cosine distance = 1 - cosine similarity
            norm_pred = norm(pred_specs, axis=-1) + 1e-22
            norm_true = norm(true_spec, axis=-1) + 1e-22
            dist.append(1 - np.dot(pred_specs, true_spec) / (norm_pred * norm_true))
        elif func == 'entropy':
            def norm_peaks(prob):
                return prob / (prob.sum(axis=-1, keepdims=True) + 1e-22)

            def entropy(prob):
                return -np.sum(prob * np.log(prob + 1e-22), axis=-1)

            norm_pred = norm_peaks(pred_specs)
            norm_true = norm_peaks(true_spec)
            zeros = (pred_specs.sum(axis=-1) == 0)
            entropy_pred = entropy(norm_pred)
            entropy_targ = entropy(norm_true)
            entropy_mix = entropy((norm_pred + norm_true) / 2)
            entropy_dists = (2 * entropy_mix - entropy_pred - entropy_targ) / np.log(4)
            entropy_dists[zeros] = 1  # mask empty spectra to be distance 1
            dist.append(entropy_dists)

    dist = np.array(dist)  # num of colli energy x number of candidates
    
    weights = (np.array(true_npeaks) >= 5) * 3 + (np.array(true_npeaks) >= 1) * 1
    weights = weights / (weights.sum() + 1e-9)
    
    if agg:
        return np.sum(dist * weights[:, None], axis=0)
    else:
        # return both
        dist = dist[weights > 0]  # exclude any objectives that have zero-weights based on filter
        return dist, np.sum(dist * weights[:, None], axis=0)


cos_dist_bin = partial(dist_bin, func='cos')


def load_true_spectrum(spec_file: Path, spec_name: str, num_bins: int = 15000, upper_limit: int = 1500) -> Dict[str, np.ndarray]:
    """
    Load ground-truth spectrum strings from spec_files.hdf5 and bin them.

    Args:
        spec_file: path to spec_files.hdf5
        spec_name: spectrum id
        num_bins: number of m/z bins
        upper_limit: max m/z

    Returns:
        dict mapping collision energy (str) to binned array
    """
    spec_h5 = common.HDF5Dataset(spec_file)
    
    try:
        spec_str = spec_h5.read_str(spec_name)
        
        spec_lines = spec_str.split('\n')
        all_spectra_tuples = []
        meta = {}
        cur_spectra = []
        current_ce = None
        
        is_mgf_format = "BEGIN IONS" in spec_str
        is_sirius_format = any(line.strip().startswith(">collision") for line in spec_lines)
        
        if is_mgf_format:
            in_ions = False
            for line in spec_lines:
                line = line.strip()
                if line == "BEGIN IONS":
                    in_ions = True
                    meta = {}
                    cur_spectra = []
                    continue
                elif line == "END IONS":
                    in_ions = False
                    if len(cur_spectra) > 0:
                        spec_data = np.vstack(cur_spectra)
                        collision_energy = meta.get('COLLISION_ENERGY', meta.get('collision_energy', '30.0'))
                        all_spectra_tuples.append((collision_energy, spec_data))
                    continue
                elif not line:
                    continue
                elif "=" in line:
                    k, v = [i.strip() for i in line.split("=", 1)]
                    meta[k] = v
                elif in_ions:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            mz = float(parts[0])
                            intens = float(parts[1])
                            cur_spectra.append((mz, intens))
                        except ValueError:
                            continue
            
            if len(all_spectra_tuples) == 0 and len(cur_spectra) > 0:
                spec_data = np.vstack(cur_spectra)
                collision_energy = meta.get('COLLISION_ENERGY', meta.get('collision_energy', '30.0'))
                all_spectra_tuples = [(collision_energy, spec_data)]
        elif is_sirius_format:
            for line in spec_lines:
                line = line.strip()
                if not line:
                    continue
                elif line.startswith("#"):
                    if "COLLISION ENERGY" in line.upper():
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part.upper() == "ENERGY":
                                if i + 1 < len(parts):
                                    try:
                                        current_ce = float(parts[i + 1])
                                    except ValueError:
                                        current_ce = 30.0
                                    break
                elif line.startswith(">collision"):
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            current_ce = float(parts[1])
                        except ValueError:
                            current_ce = 30.0
                    if len(cur_spectra) > 0 and current_ce is not None:
                        spec_data = np.vstack(cur_spectra)
                        all_spectra_tuples.append((str(current_ce), spec_data))
                        cur_spectra = []
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            mz = float(parts[0])
                            intens = float(parts[1])
                            cur_spectra.append((mz, intens))
                        except ValueError:
                            continue
            
            if len(cur_spectra) > 0:
                spec_data = np.vstack(cur_spectra)
                ce = str(current_ce) if current_ce is not None else '30.0'
                all_spectra_tuples.append((ce, spec_data))
            
            if len(all_spectra_tuples) == 0 and len(cur_spectra) > 0:
                spec_data = np.vstack(cur_spectra)
                ce = str(current_ce) if current_ce is not None else '30.0'
                all_spectra_tuples = [(ce, spec_data)]
        else:
            for line in spec_lines:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith(">"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        mz = float(parts[0])
                        intens = float(parts[1])
                        cur_spectra.append((mz, intens))
                    except ValueError:
                        continue
            
            if len(cur_spectra) > 0:
                spec_data = np.vstack(cur_spectra)
                all_spectra_tuples = [('30.0', spec_data)]
        
        spectra_tuples = all_spectra_tuples
        
        spec_h5.close()
        
        true_spec_dict = {}
        for ce, spec_data in spectra_tuples:
            if len(spec_data) == 0:
                continue
            
            if isinstance(ce, str):
                ce_clean = ce.replace(' eV', '').replace(' eV [imputed]', '').strip()
                try:
                    ce_float = float(ce_clean)
                except ValueError:
                    ce_float = 30.0
            else:
                ce_float = float(ce)
            
            ce_key = f'{ce_float:.0f}'
            
            binned = common.bin_spectra([spec_data], num_bins=num_bins, upper_limit=upper_limit)[0]
            true_spec_dict[ce_key] = binned
        
        return true_spec_dict
    except Exception as e:
        spec_h5.close()
        return {}


def load_predicted_spectrum(pred_file: Path, spec_name: str, num_bins: int = 15000, upper_limit: int = 1500) -> Dict[str, np.ndarray]:
    """
    Load predicted spectra from preds.hdf5.

    Layout: pred_<spec_name>/ikey <inchikey>/collision <energy>/spec

    Args:
        pred_file: predictions HDF5
        spec_name: spectrum id
        num_bins: number of m/z bins
        upper_limit: max m/z

    Returns:
        dict mapping collision energy (str) to binned array
    """
    with h5py.File(pred_file, 'r') as f:
        pred_key = f'pred_{spec_name}'
        
        if pred_key not in f:
            return {}
        
        pred_group = f[pred_key]
        pred_dict = {}
        
        sparse_out = f.attrs.get('sparse_out', True)
        
        for ikey_key in pred_group.keys():
            ikey_group = pred_group[ikey_key]
            
            for ce_key in ikey_group.keys():
                ce_group = ikey_group[ce_key]
                if 'spec' in ce_group:
                    spec_data = ce_group['spec'][:]
                    
                    ce_raw = ce_key.replace('collision ', '').strip()
                    
                    try:
                        ce_float = float(ce_raw)
                    except ValueError:
                        ce_float = 30.0
                    
                    ce_key_normalized = f'{ce_float:.0f}'
                    
                    if sparse_out and len(spec_data.shape) == 2 and spec_data.shape[1] >= 2:
                        binned = common.bin_spectra([spec_data], num_bins=num_bins, upper_limit=upper_limit)[0]
                        pred_dict[ce_key_normalized] = binned
                    elif len(spec_data.shape) == 1:
                        pred_dict[ce_key_normalized] = spec_data
                    else:
                        if len(spec_data.shape) == 2:
                            binned = common.bin_spectra([spec_data], num_bins=num_bins, upper_limit=upper_limit)[0]
                            pred_dict[ce_key_normalized] = binned
                        else:
                            pred_dict[ce_key_normalized] = np.zeros(num_bins)
        
        return pred_dict


def main():
    parser = argparse.ArgumentParser(description='Evaluate test-set cosine similarity from preds.hdf5')
    parser.add_argument('--gen-checkpoint', type=str, default=None, help='Generation checkpoint (optional if --pred-file set)')
    parser.add_argument('--inten-checkpoint', type=str, default=None, help='Intensity checkpoint (optional if --pred-file set)')
    parser.add_argument('--pred-file', type=str, default=None, help='Predictions HDF5 (preds.hdf5); skips lookup under output-dir/predictions')
    parser.add_argument('--dataset-name', type=str, default='new_dataset', help='Dataset folder under data/foriceberg/spec_datasets/')
    parser.add_argument('--split-name', type=str, default='split_1.tsv', help='Split TSV under dataset/splits/')
    parser.add_argument('--output-dir', type=str, default='test/evaluation', help='Output directory for evaluation_results.tsv')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size (hint for predictor command)')
    parser.add_argument('--num-workers', type=int, default=0, help='Num workers (hint for predictor command)')
    parser.add_argument('--gpu', action='store_true', help='Use GPU (hint for predictor command)')
    parser.add_argument('--num-bins', type=int, default=15000, help='Number of m/z bins')
    parser.add_argument('--upper-limit', type=int, default=1500, help='Upper m/z limit')
    parser.add_argument('--ignore-parent-peak', action='store_true', help='Zero precursor bin in cosine')
    parser.add_argument('--test-labels', type=str, default=None, help='Test labels TSV (overrides split-based test set)')
    
    args = parser.parse_args()
    
    if args.pred_file is None and (args.gen_checkpoint is None or args.inten_checkpoint is None):
        parser.error('Provide --pred-file or both --gen-checkpoint and --inten-checkpoint')
    
    project_root = repo_root()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("ICEBERG test-set evaluation")
    print("=" * 80)
    if args.gen_checkpoint:
        print(f"Gen checkpoint: {args.gen_checkpoint}")
    if args.inten_checkpoint:
        print(f"Inten checkpoint: {args.inten_checkpoint}")
    if args.pred_file:
        print(f"Predictions: {args.pred_file}")
    print(f"Dataset: {args.dataset_name}")
    if args.test_labels:
        print(f"Test labels: {args.test_labels}")
    else:
        print(f"Split file: {args.split_name}")
    print("=" * 80)
    
    dataset_path = data_spec_datasets(args.dataset_name)
    
    dataset_labels_file = dataset_path / "labels.tsv"
    dataset_labels = None
    if dataset_labels_file.exists():
        dataset_labels = pd.read_csv(dataset_labels_file, sep='\t')
        dataset_labels_dict = dict(zip(dataset_labels['spec'], dataset_labels.to_dict('records')))
    else:
        dataset_labels_dict = {}
    
    if args.test_labels:
        test_labels_file = Path(args.test_labels)
        if not test_labels_file.exists():
            raise FileNotFoundError(f"Test labels file not found: {test_labels_file}")
        test_labels = pd.read_csv(test_labels_file, sep='\t')
        print(f"\nLoaded test set from file: {len(test_labels)} samples")
    else:
        labels_file = dataset_path / "labels.tsv"
        split_file = dataset_path / "splits" / args.split_name
        
        if not labels_file.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_file}")
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        labels_df = pd.read_csv(labels_file, sep='\t')
        spec_names = labels_df['spec'].tolist()
        train_inds, val_inds, test_inds = get_splits(spec_names, str(split_file))
        test_indices = test_inds
        
        print(f"\nTest set size: {len(test_indices)} samples")
        
        test_labels = labels_df.iloc[test_indices].copy()
        test_labels_file = output_dir / "test_labels.tsv"
    test_labels.to_csv(test_labels_file, sep='\t', index=False)
    
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    
    if args.pred_file:
        pred_file = Path(args.pred_file)
        if not pred_file.exists():
            raise FileNotFoundError(f"Predictions file not found: {pred_file}")
        print(f"\nUsing predictions file: {pred_file}")
    else:
        pred_files = list(pred_dir.glob("preds*.hdf5"))
        
        if not pred_files:
            print("\nNo predictions file found")
            print("=" * 80)
            print("Run prediction first, for example:")
            print("=" * 80)
            print(f"python src/ms_pred/dag_pred/predict_smis.py \\")
            print(f"    --gen-checkpoint {args.gen_checkpoint} \\")
            print(f"    --inten-checkpoint {args.inten_checkpoint} \\")
            print(f"    --dataset-labels {test_labels_file} \\")
            print(f"    --batch-size {args.batch_size} \\")
            print(f"    --num-workers {args.num_workers} \\")
            print(f"    --sparse-out \\")
            print(f"    --sparse-k 100 \\")
            print(f"    --max-nodes 100 \\")
            print(f"    --threshold 0.1 \\")
            print(f"    --save-dir {pred_dir} \\")
            print(f"    --adduct-shift" + (" \\" if args.gpu else ""))
            if args.gpu:
                print(f"    --gpu")
            print("=" * 80)
            print("Then re-run this script to compute cosine similarity")
            return
        
        pred_file = pred_files[0]
        print(f"\nFound predictions: {pred_file}")
    
    spec_file = dataset_path / "spec_files.hdf5"
    if not spec_file.exists():
        print(f"ERROR: spec_files.hdf5 not found: {spec_file}")
        return
    
    print("\nComputing cosine similarity...")
    
    with h5py.File(pred_file, 'r') as f:
        num_bins = f.attrs.get('num_bins', args.num_bins)
        upper_limit = f.attrs.get('upper_limit', args.upper_limit)
    
    results = []
    
    partial_results_file = output_dir / "evaluation_results_partial.tsv"
    processed_specs = set()
    if partial_results_file.exists():
        print(f"\nResuming from partial results: {partial_results_file}")
        try:
            partial_df = pd.read_csv(partial_results_file, sep='\t')
            processed_specs = set(partial_df['spec'].tolist())
            results = partial_df.to_dict('records')
            print(f"Already processed {len(processed_specs)} samples; continuing...")
        except Exception as e:
            print(f"Could not read partial results ({e}); starting fresh")
    
    iterator = tqdm(test_labels.itertuples(), total=len(test_labels), desc="Cosine similarity") if USE_TQDM else test_labels.itertuples()
    
    processed_count = 0
    for row in iterator:
        spec_name = row.spec
        
        if spec_name in processed_specs:
            continue

        precursor_mass = None
        if hasattr(row, 'precursor') and pd.notna(row.precursor):
            precursor_mass = row.precursor
        elif spec_name in dataset_labels_dict:
            spec_info = dataset_labels_dict[spec_name]
            if 'precursor' in spec_info and pd.notna(spec_info['precursor']):
                precursor_mass = spec_info['precursor']
        
        if precursor_mass is None:
            try:
                from ms_pred.common import chem_utils
                smiles = None
                adduct = None
                
                if hasattr(row, 'smiles') and pd.notna(row.smiles):
                    smiles = row.smiles
                elif spec_name in dataset_labels_dict and 'smiles' in dataset_labels_dict[spec_name]:
                    smiles = dataset_labels_dict[spec_name]['smiles']
                
                if hasattr(row, 'ionization') and pd.notna(row.ionization):
                    adduct = row.ionization
                elif spec_name in dataset_labels_dict and 'ionization' in dataset_labels_dict[spec_name]:
                    adduct = dataset_labels_dict[spec_name]['ionization']
                
                if smiles:
                    precursor_mass = chem_utils.mass_from_smi(smiles)
                    if adduct:
                        precursor_mass += common.ion2mass.get(adduct, 0)
            except Exception as e:
                print(f"Warning: could not compute precursor mass for {spec_name}: {e}")
                precursor_mass = None
        
        if precursor_mass is None:
            print(f"Warning: missing precursor mass for {spec_name}; skip")
            continue
        
        true_spec_dict = load_true_spectrum(spec_file, spec_name, num_bins=num_bins, upper_limit=upper_limit)
        
        if not true_spec_dict:
            print(f"Warning: could not load true spectrum {spec_name}")
            continue
        
        pred_spec_dict = load_predicted_spectrum(pred_file, spec_name, num_bins=num_bins, upper_limit=upper_limit)
        
        if not pred_spec_dict:
            print(f"Warning: could not load predicted spectrum {spec_name}")
            continue
        
        pred_list = [pred_spec_dict]
        
        ignore_peak = None
        if args.ignore_parent_peak:
            ignore_peak = int((precursor_mass - 1) * num_bins / upper_limit)
        
        try:
            cosine_dist = cos_dist_bin(
                cand_preds_dict=pred_list,
                true_spec_dict=true_spec_dict,
                sparse=False,
                ignore_peak=ignore_peak,
                agg=True
            )
            
            # cosine similarity = 1 - cosine distance
            cosine_sim = 1 - cosine_dist[0]
            
            smiles = getattr(row, 'smiles', 'N/A')
            try:
                if smiles != 'N/A':
                    inchikey = common.inchikey_from_smiles(smiles)
                else:
                    inchikey = 'N/A'
            except:
                inchikey = 'N/A'
            
            results.append({
                'spec': spec_name,
                'smiles': smiles,
                'inchikey': inchikey,
                'precursor_mass': precursor_mass,
                'cosine_distance': float(cosine_dist[0]),
                'cosine_similarity': float(cosine_sim),
            })
            processed_count += 1
            
            if processed_count % 100 == 0:
                results_df = pd.DataFrame(results)
                results_df.to_csv(partial_results_file, sep='\t', index=False)
                if USE_TQDM:
                    iterator.set_postfix({'saved': len(results)})
        except Exception as e:
            print(f"\nWarning: cosine similarity failed for {spec_name}: {e}")
            continue
    
    if not results:
        print("ERROR: no successful evaluations")
        return
    
    results_df = pd.DataFrame(results)
    results_file = output_dir / "evaluation_results.tsv"
    results_df.to_csv(results_file, sep='\t', index=False)
    print(f"\nWrote results: {results_file}")
    
    cosine_sims = results_df['cosine_similarity'].values
    
    stats = {
        'num_samples': len(results),
        'mean_cosine_similarity': float(np.mean(cosine_sims)),
        'std_cosine_similarity': float(np.std(cosine_sims)),
        'median_cosine_similarity': float(np.median(cosine_sims)),
        'min_cosine_similarity': float(np.min(cosine_sims)),
        'max_cosine_similarity': float(np.max(cosine_sims)),
        'q25_cosine_similarity': float(np.percentile(cosine_sims, 25)),
        'q75_cosine_similarity': float(np.percentile(cosine_sims, 75)),
    }
    
    print("\n" + "=" * 80)
    print("Evaluation summary")
    print("=" * 80)
    print(f"Samples: {stats['num_samples']}")
    print(f"Mean cosine similarity: {stats['mean_cosine_similarity']:.4f}")
    print(f"Std dev: {stats['std_cosine_similarity']:.4f}")
    print(f"Median: {stats['median_cosine_similarity']:.4f}")
    print(f"Min: {stats['min_cosine_similarity']:.4f}")
    print(f"Max: {stats['max_cosine_similarity']:.4f}")
    print(f"Q25: {stats['q25_cosine_similarity']:.4f}")
    print(f"Q75: {stats['q75_cosine_similarity']:.4f}")
    print("=" * 80)
    
    stats_file = output_dir / "evaluation_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nWrote stats: {stats_file}")


if __name__ == "__main__":
    main()
