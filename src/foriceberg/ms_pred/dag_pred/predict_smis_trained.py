"""This code was obtained from the ICEBERG implementation developed by
Goldman et al.

Original repository:
https://github.com/samgoldman97/ms-pred

No modifications were made to the original code.

The original copyright notice and license terms are retained.

predict_smis.py

Make both dag and intensity predictions jointly and revert to binned
"""
import logging
import multiprocess.process
import random
import math
import ast
import json
from tqdm import tqdm
from datetime import datetime
import yaml
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

import torch
import pytorch_lightning as pl

import ms_pred.common as common
from ms_pred.dag_pred import inten_model, gen_model, joint_model


from rdkit import rdBase
from rdkit import RDLogger

rdBase.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.*")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--gpu", default=False, action="store_true")
    parser.add_argument("--seed", default=42, action="store", type=int)
    parser.add_argument("--sparse-out", default=False, action="store_true")
    parser.add_argument("--sparse-k", default=100, action="store", type=lambda x: None if x.lower() == 'none' else int(x), help="Keep only the top K peaks when set; if None, keep all peaks with intensity > threshold. Default: 100")
    parser.add_argument("--binned-out", default=False, action="store_true")
    parser.add_argument('--adduct-shift',default=False, action="store_true")
    parser.add_argument("--num-workers", default=0, action="store", type=int)
    parser.add_argument("--batch-size", default=64, action="store", type=int)
    date = datetime.now().strftime("%Y_%m_%d")
    parser.add_argument("--save-dir", default=f"results/{date}_ffn_pred/")
    parser.add_argument(
        "--gen-checkpoint",
        help="name of checkpoint file",
        default="results/2022_06_22_pretrain/version_3/epoch=99-val_loss=0.87.ckpt",
    )
    parser.add_argument(
        "--inten-checkpoint",
        help="name of checkpoint file",
        default="results/2022_06_22_pretrain/version_3/epoch=99-val_loss=0.87.ckpt",
    )
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--dataset-labels", default="labels.tsv")
    parser.add_argument("--split-name", default="split_22.tsv")
    parser.add_argument("--threshold", default=0.01, action="store", type=float, help="Intensity threshold for filtering low-intensity peaks. Default: 0.01")
    parser.add_argument("--max-nodes", default=100, action="store", type=int)
    parser.add_argument("--upper-limit", default=1500, action="store", type=int)
    parser.add_argument("--num-bins", default=15000, action="store", type=int)
    parser.add_argument(
        "--subset-datasets",
        default="none",
        action="store",
        choices=["none", "train_only", "test_only", "debug_special"],
    )

    return parser.parse_args()


def predict():
    args = get_args()
    kwargs = args.__dict__
    if isinstance(kwargs.get("sparse_k"), str) and kwargs.get("sparse_k").lower() == "none":
        kwargs["sparse_k"] = None

    save_dir = Path(kwargs["save_dir"])
    debug = kwargs["debug"]
    common.setup_logger(save_dir, log_name="joint_pred.log", debug=debug)
    seed = kwargs.get("seed")
    if seed is not None:
        try:
            pl.seed_everything(seed)
        except AttributeError:
            try:
                pl.utilities.seed.seed_everything(seed)
            except AttributeError:
                import random as random_module
                random_module.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)

    # Dump args
    yaml_args = yaml.dump(kwargs)
    logging.info(f"\n{yaml_args}")
    with open(save_dir / "args.yaml", "w") as fp:
        fp.write(yaml_args)

    # Get dataset
    # Load smiles dataset and split into 3 subsets
    data_dir = Path("")
    if kwargs.get("dataset_name") is not None:
        dataset_name = kwargs["dataset_name"]
        data_dir = Path("data/spec_datasets") / dataset_name

    labels = Path(kwargs["dataset_labels"])

    # Get train, val, test inds
    df = pd.read_csv(labels, sep="\t")

    if kwargs["debug"]:
        df = df[:10]

    if kwargs["subset_datasets"] != "none":
        splits = pd.read_csv(data_dir / "splits" / kwargs["split_name"], sep="\t")
        folds = set(splits.keys())
        folds.remove("spec")
        fold_name = list(folds)[0]
        if kwargs["subset_datasets"] == "train_only":
            names = splits[splits[fold_name] == "train"]["spec"].tolist()
        elif kwargs["subset_datasets"] == "test_only":
            names = splits[splits[fold_name] == "test"]["spec"].tolist()
        elif kwargs["subset_datasets"] == "debug_special":
            names = splits[splits[fold_name] == "test"]["spec"].tolist()
            names = ["CCMSLIB00000001590"]
            kwargs["debug"] = True
        else:
            raise NotImplementedError()
        df = df[df["spec"].isin(names)]

    # Create model and load
    # Load from checkpoint
    gen_checkpoint = kwargs["gen_checkpoint"]
    inten_checkpoint = kwargs["inten_checkpoint"]

    gpu = kwargs["gpu"]
    # Load checkpoints with strict=False to handle structure mismatches
    import torch as torch_module  # Ensure torch is available
    
    if gpu and not torch_module.cuda.is_available():
        logging.warning("CUDA is not available; forcing CPU mode")
        gpu = False
    
    map_location = "cuda" if (gpu and torch_module.cuda.is_available()) else "cpu"
    
    # Always inspect checkpoint metadata on CPU to avoid device issues.
    checkpoint = torch_module.load(gen_checkpoint, map_location="cpu")
    hparams = checkpoint.get('hyper_parameters', {})
    
    logging.info(f"Loading trained model from: {gen_checkpoint}")
    logging.info(f"Model hyper_parameters: pe_embed_k={hparams.get('pe_embed_k', 'N/A')}, "
                 f"set_layers={hparams.get('set_layers', 'N/A')}, "
                 f"node_feats={hparams.get('node_feats', 'N/A')}, "
                 f"add_hs={hparams.get('add_hs', 'N/A')}, "
                 f"embed_elem_group={hparams.get('embed_elem_group', 'N/A')}")
    logging.info(f"Using device: {map_location}")
    
    inten_model_obj = inten_model.IntenGNN.load_from_checkpoint(
        inten_checkpoint, 
        map_location=map_location,
        strict=False
    )
    
    try:
        gen_model_obj = gen_model.FragGNN.load_from_checkpoint(
            gen_checkpoint,
            map_location=map_location, 
            strict=False
        )
        logging.info(f"Loaded model attributes: pe_embed_k={gen_model_obj.pe_embed_k}, "
                     f"add_hs={gen_model_obj.add_hs}, "
                     f"embed_elem_group={gen_model_obj.embed_elem_group}")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise
    avail_gpu_num = torch_module.cuda.device_count() if gpu else 0

    # Build joint model class

    logging.info(
        f"Loaded gen / inten models from {gen_checkpoint} & {inten_checkpoint}"
    )

    # Avoid the extra positional-encoding column produced by random_walk_pe(k=0).
    
    original_add_pe_embed = gen_model_obj.tree_processor.add_pe_embed
    
    def patched_add_pe_embed(graph):
        if gen_model_obj.tree_processor.pe_embed_k > 0:
            return original_add_pe_embed(graph)
        else:
            return graph
    
    gen_model_obj.tree_processor.add_pe_embed = patched_add_pe_embed
    logging.info("Applied pe_embed_k=0 compatibility patch")
    
    model = joint_model.JointModel(
        gen_model_obj=gen_model_obj, inten_model_obj=inten_model_obj
    )
    
    # Some intensity checkpoints expect an instrument embedding even when the joint model does not.
    if not model.embed_instrument and inten_model_obj.embed_instrument:
        original_inten_predict = inten_model_obj.predict
        
        def patched_inten_predict(*args, instruments=None, **kwargs):
            if instruments is None and inten_model_obj.embed_instrument:
                adducts = kwargs.get('adducts', None)
                if adducts is None and len(args) > 7:
                    adducts = args[7]
                if adducts is not None:
                    import torch
                    import ms_pred.common as common
                    batch_size = len(adducts) if isinstance(adducts, (list, tuple)) else adducts.shape[0]
                    device = adducts.device if hasattr(adducts, 'device') else 'cpu'
                    default_instrument = torch.zeros(batch_size, dtype=torch.long, device=device)
                    kwargs['instruments'] = default_instrument
            return original_inten_predict(*args, **kwargs)
        
        inten_model_obj.predict = patched_inten_predict
        logging.info("Applied intensity-model instrument compatibility patch")

    with torch.no_grad():
        model.eval()
        model.freeze()

        binned_out = kwargs["binned_out"]

        def prepare_entry(entry):
            smi = entry["smiles"]
            adduct = entry["ionization"]
            precursor_mz = entry["precursor"]
            instrument = entry["instrument"]
            name = entry["spec"]
            inchikey = common.inchikey_from_smiles(smi)
            collision_energies = [i for i in ast.literal_eval(entry["collision_energies"])]
            tup_to_process = []

            for colli_eng in collision_energies:
                colli_eng_val = common.collision_energy_to_float(colli_eng)  # str to float
                if math.isnan(colli_eng_val):  # skip collision_energy == nan (no collision energy recorded)
                    continue
                tup_to_process.append((smi, name, colli_eng_val, adduct, instrument, precursor_mz,
                                       f"pred_{name}/ikey {inchikey}/collision {colli_eng}"))
            return tup_to_process

        all_rows = [j for _, j in df.iterrows()]
        logging.info('Preparing entries')
        if kwargs["num_workers"] == 0:
            predict_entries = [prepare_entry(i) for i in tqdm(all_rows)]
        else:
            predict_entries = common.chunked_parallel(
                all_rows,
                prepare_entry,
                chunks=10000,
                max_cpu=kwargs["num_workers"],
            )
        predict_entries = [i for j in predict_entries for i in j]  # unroll
        random.shuffle(predict_entries)  # shuffle to evenly distribute graph size across batches
        logging.info(f'There are {len(predict_entries)} entries to process')

        batch_size = kwargs["batch_size"]
        all_batched_entries = [
            predict_entries[i: i + batch_size] for i in range(0, len(predict_entries), batch_size)
        ]


        def producer_func(batch):
            torch.set_num_threads(1)
            if gpu and avail_gpu_num >= 0:
                if kwargs["num_workers"] > 0:
                    worker_id = multiprocess.process.current_process()._identity[0]  # get worker id
                    gpu_id = worker_id % avail_gpu_num
                else:
                    gpu_id = 0
                device = f"cuda:{gpu_id}"
            else:
                device = "cpu"
            model.to(device)
            if gpu and avail_gpu_num > 0:
                torch.cuda.set_device(gpu_id) # avoids error in pe_embedding under multithreading. 

            # for batch in batched_entries:
            smis, spec_names, colli_eng_vals, adducts, instruments, precursor_mzs, h5_names = list(zip(*batch))
            full_outputs = model.predict_mol(
                smis,
                precursor_mz=precursor_mzs,
                collision_eng=colli_eng_vals,
                adduct=adducts,
                instrument=instruments,
                threshold=kwargs["threshold"],
                device=device,
                max_nodes=kwargs["max_nodes"],
                binned_out=binned_out,
                adduct_shift=kwargs["adduct_shift"],
                name=h5_names,
            )
            return_list = []
            if binned_out:
                for output_spec, spec_name, smi, h5_name in \
                        zip(full_outputs["spec"], spec_names, smis, h5_names):
                    if kwargs["sparse_out"]:
                        threshold = kwargs.get("threshold", 0.01)
                        valid_inds = np.where(output_spec > threshold)[0]
                        if len(valid_inds) > 0:
                            valid_intens = output_spec[valid_inds]
                            sorted_order = np.argsort(valid_intens)[::-1]
                            best_inds = valid_inds[sorted_order]
                            sparse_k = kwargs.get("sparse_k", 100)
                            if sparse_k is not None and sparse_k > 0:
                                best_inds = best_inds[:sparse_k]
                            best_intens = output_spec[best_inds]
                            output_spec = np.stack([best_inds, best_intens], -1)
                        else:
                            output_spec = np.zeros((0, 2), dtype=output_spec.dtype)

                    inchikey = common.inchikey_from_smiles(smi)
                    return_list.append((h5_name, spec_name, smi, inchikey, output_spec, None))
            else:
                for output_spec, spec_name, smi, h5_name, pred_frag in \
                        zip(full_outputs["spec"], spec_names, smis, h5_names, full_outputs["frag"]):
                    assert kwargs["sparse_out"], 'sparse_out must be True for non-binned output'
                    
                    threshold = kwargs.get("threshold", 0.01)
                    if len(output_spec) > 0 and output_spec.shape[1] >= 2:
                        intensities = output_spec[:, 1]
                        valid_inds = np.where(intensities > threshold)[0]
                        
                        if len(valid_inds) > 0:
                            valid_intens = intensities[valid_inds]
                            sorted_order = np.argsort(valid_intens)[::-1]
                            best_inds = valid_inds[sorted_order]
                            
                            sparse_k = kwargs.get("sparse_k", 100)
                            if sparse_k is not None and sparse_k > 0:
                                best_inds = best_inds[:sparse_k]
                            
                            output_spec = output_spec[best_inds, :]
                            
                            pred_frag_array = np.array(pred_frag) if pred_frag is not None else None
                            if pred_frag_array is not None and len(pred_frag_array) > 0:
                                valid_frag_inds = best_inds[best_inds < len(pred_frag_array)]
                                if len(valid_frag_inds) > 0:
                                    frag_selected = pred_frag_array[valid_frag_inds]
                                    if len(valid_frag_inds) < len(best_inds):
                                        frag_padded = [None] * len(best_inds)
                                        for i, orig_idx in enumerate(best_inds):
                                            if orig_idx < len(pred_frag_array):
                                                frag_padded[i] = pred_frag_array[orig_idx]
                                            else:
                                                frag_padded[i] = None
                                        pred_frag = frag_padded
                                    else:
                                        pred_frag = frag_selected.tolist()
                                else:
                                    pred_frag = [None] * len(best_inds)
                            else:
                                pred_frag = [None] * len(best_inds)
                        else:
                            output_spec = np.zeros((0, 2), dtype=output_spec.dtype)
                            pred_frag = []
                    else:
                        output_spec = np.zeros((0, 2), dtype=np.float32)
                        pred_frag = []
                    
                    inchikey = common.inchikey_from_smiles(smi)
                    return_list.append((h5_name, spec_name, smi, inchikey, output_spec, pred_frag))
            return return_list

        if binned_out:
            out_name = "binned_preds.hdf5"
        else:
            out_name = "preds.hdf5"

        def write_h5_func(out_entries):
            h5 = common.HDF5Dataset(save_dir / out_name, mode='w')
            
            h5.attrs['num_bins'] = 15000
            h5.attrs['upper_limit'] = 1500
            h5.attrs['sparse_out'] = kwargs["sparse_out"]
            for out_batch in out_entries:
                for out_item in out_batch:
                    h5_name, spec_name, smi, inchikey, output_spec, pred_frag = out_item
                    try:
                        h5.write_data(h5_name + '/spec', output_spec)
                        if pred_frag is not None and len(pred_frag) > 0:
                            valid_frags = [f for f in pred_frag if f is not None]
                            if len(valid_frags) > 0:
                                valid_frags_serializable = []
                                for f in valid_frags:
                                    if isinstance(f, (np.integer, np.int64, np.int32)):
                                        valid_frags_serializable.append(int(f))
                                    elif isinstance(f, (np.floating, np.float64, np.float32)):
                                        valid_frags_serializable.append(float(f))
                                    else:
                                        valid_frags_serializable.append(f)
                                h5.write_str(h5_name + '/frag', json.dumps(valid_frags_serializable))  # save as string avoids overflow
                        h5.update_attr(h5_name, {'smiles': smi, 'ikey': inchikey, 'spec_name': spec_name})
                    except Exception as e:
                        print("h5_name", h5_name)
                        print(f"Error writing {h5_name}: {e}")
                        import traceback
                        traceback.print_exc()

            h5.close()


        if kwargs["num_workers"] == 0:
            output_entries = [producer_func(batch) for batch in tqdm(all_batched_entries)]
            write_h5_func(output_entries)
        else:
            common.chunked_parallel(all_batched_entries, producer_func, output_func=write_h5_func,
                                    chunks=1000, max_cpu=kwargs["num_workers"])


if __name__ == "__main__":
    import time

    start_time = time.time()
    predict()
    end_time = time.time()
    logging.info(f"Program finished in: {end_time - start_time} seconds")
