"""predict_smis_trained_inten.py

Predict spectra with trained generation and intensity checkpoints.
Based on predict_smis_trained.py, but uses a user-trained intensity model.

"""
import sys
from pathlib import Path

_src_root = Path(__file__).resolve().parent.parent.parent  # .../src (parent of package `ms_pred`)
if str(_src_root) not in sys.path:
    sys.path.insert(0, str(_src_root))

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
        default="models/pretrained/iceberg_dag_inten_msg_best.ckpt",
    )
    parser.add_argument("--dataset-labels", default="data/spec_datasets/new_dataset/labels.tsv")
    parser.add_argument("--max-nodes", default=15000, type=int)
    parser.add_argument("--threshold", default=0.01, type=float)
    # When some molecules contain elements not supported by MAGMA (e.g. KeyError: 'Sn'),
    # skip those entries instead of crashing the whole prediction run.
    parser.add_argument("--no-skip-bad-molecules", default=False, action="store_true")
    parser.add_argument("--skipped-csv-name", default="skipped_entries.csv", type=str)
    return parser.parse_args()


def predict():
    """predict."""
    args = get_args()
    kwargs = args.__dict__
    kwargs["threshold"] = float(kwargs["threshold"])

    # Set seed
    seed = kwargs.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # PyTorch Lightning seed compatibility
    try:
        pl.seed_everything(seed)
    except AttributeError:
        try:
            pl.utilities.seed.seed_everything(seed)
        except AttributeError:
            pass

    gen_checkpoint = kwargs["gen_checkpoint"]
    inten_checkpoint = kwargs["inten_checkpoint"]
    save_dir = Path(kwargs["save_dir"])
    save_dir.mkdir(exist_ok=True, parents=True)

    # Save args
    yaml_args = yaml.dump(kwargs, default_flow_style=False)
    (save_dir / "args.yaml").write_text(yaml_args)

    logging.basicConfig(
        level=logging.INFO if not kwargs["debug"] else logging.DEBUG,
        format="%(asctime)s.%(msecs)03d [%(levelname)s]  %(message)s",
        datefmt="%y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(save_dir / "joint_pred.log"),
            logging.StreamHandler(),
        ],
    )

    logging.info(f"Args are {kwargs}")

    skip_bad_molecules = not kwargs.get("no_skip_bad_molecules", False)
    skipped_csv_name = kwargs.get("skipped_csv_name", "skipped_entries.csv") if "skipped_csv_name" in kwargs else kwargs.get(
        "skipped-csv-name", "skipped_entries.csv"
    )
    # Store only in main process; wrapper script uses --num-workers 0 to ensure deterministic behavior.
    skipped_records = []

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
    
    logging.info(f"Loading trained generation model from: {gen_checkpoint}")
    logging.info(f"Model hyper_parameters: pe_embed_k={hparams.get('pe_embed_k', 'N/A')}, "
                 f"set_layers={hparams.get('set_layers', 'N/A')}, "
                 f"node_feats={hparams.get('node_feats', 'N/A')}, "
                 f"add_hs={hparams.get('add_hs', 'N/A')}, "
                 f"embed_elem_group={hparams.get('embed_elem_group', 'N/A')}")
    logging.info(f"Using device: {map_location}")
    
    logging.info(f"Loading trained intensity model from: {inten_checkpoint}")
    inten_model_obj = inten_model.IntenGNN.load_from_checkpoint(
        inten_checkpoint, 
        map_location=map_location,
        strict=False
    )
    
    # Older checkpoints may contain isomer_attn_out when the current model does not.
    checkpoint_state = torch_module.load(inten_checkpoint, map_location="cpu")
    if 'isomer_attn_out.weight' in checkpoint_state.get('state_dict', {}):
        if not hasattr(inten_model_obj, 'isomer_attn_out'):
            logging.info("Checkpoint contains isomer_attn_out; adding the missing module")
            import copy
            inten_model_obj.isomer_attn_out = copy.deepcopy(inten_model_obj.output_map)
            inten_model_obj.isomer_attn_out.load_state_dict({
                'weight': checkpoint_state['state_dict']['isomer_attn_out.weight'],
                'bias': checkpoint_state['state_dict']['isomer_attn_out.bias']
            })
            logging.info("Loaded isomer_attn_out weights")
        else:
            try:
                inten_model_obj.isomer_attn_out.load_state_dict({
                    'weight': checkpoint_state['state_dict']['isomer_attn_out.weight'],
                    'bias': checkpoint_state['state_dict']['isomer_attn_out.bias']
                }, strict=True)
            except:
                pass
    
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

    avail_gpu_num = torch_module.cuda.device_count() if (gpu and torch_module.cuda.is_available()) else 0

    # Build joint model class

    logging.info(
        f"Loaded gen / inten models from {gen_checkpoint} & {inten_checkpoint}"
    )

    with torch.no_grad():
        model.eval()
        model.freeze()

        binned_out = kwargs["binned_out"]

        def prepare_entry(entry):
            smi = entry["smiles"]
            name = entry["spec"]
            colli_eng_vals = entry.get("collision_energies", "['30.0']")
            if isinstance(colli_eng_vals, str):
                colli_eng_vals = ast.literal_eval(colli_eng_vals)
            adduct = entry.get("ionization", "[M+H]+")
            instrument = entry.get("instrument", "Orbitrap")
            if instrument is None:
                instrument = "Orbitrap"
            precursor_mz = entry.get("precursor", None)
            if precursor_mz is None:
                from ms_pred.common import chem_utils
                precursor_mz = chem_utils.mass_from_smi(smi)
                if adduct:
                    precursor_mz += common.ion2mass[adduct]
            
            inchikey = common.inchikey_from_smiles(smi)
            tup_to_process = []
            for colli_eng in colli_eng_vals:
                colli_eng_val = float(colli_eng)
                tup_to_process.append((smi, name, colli_eng_val, adduct, instrument, precursor_mz,
                                       f"pred_{name}/ikey {inchikey}/collision {colli_eng}"))
            return tup_to_process

        df = pd.read_csv(kwargs["dataset_labels"], sep="\t")
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

            def _predict_one(idx: int):
                # Predict with a single entry to isolate failures to a specific molecule.
                return model.predict_mol(
                    [smis[idx]],
                    precursor_mz=[precursor_mzs[idx]],
                    collision_eng=[colli_eng_vals[idx]],
                    adduct=[adducts[idx]],
                    instrument=[instruments[idx]],
                    threshold=kwargs["threshold"],
                    device=device,
                    max_nodes=kwargs["max_nodes"],
                    binned_out=binned_out,
                    adduct_shift=kwargs["adduct_shift"],
                    name=[h5_names[idx]],
                )

            try:
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
            except Exception as e:
                if not skip_bad_molecules:
                    raise

                logging.warning(
                    f"Batch predict failed; will fallback to per-entry prediction. Error: {type(e).__name__}: {e}"
                )

                # Fallback: predict each entry individually; skip the ones that still fail.
                return_list = []
                for i in range(len(batch)):
                    try:
                        single_outputs = _predict_one(i)
                        # single_outputs values are lists with length 1
                        if binned_out:
                            for output_spec, spec_name, smi, h5_name in zip(
                                single_outputs["spec"], [spec_names[i]], [smis[i]], [h5_names[i]]
                            ):
                                if kwargs["sparse_out"]:
                                    threshold = kwargs.get("threshold", 0.01)
                                    valid_inds = np.where(output_spec > threshold)[0]
                                    if len(valid_inds) > 0:
                                        if kwargs["sparse_k"] is not None:
                                            top_k = min(kwargs["sparse_k"], len(valid_inds))
                                            top_k_inds = np.argsort(output_spec[valid_inds])[-top_k:][::-1]
                                            valid_inds = valid_inds[top_k_inds]
                                        mz_vals = valid_inds * common.BIN_SIZE + common.MIN_MASS
                                        inten_vals = output_spec[valid_inds]
                                        return_list.append(
                                            {
                                                "spec_name": spec_name,
                                                "smiles": smi,
                                                "mz": mz_vals.tolist(),
                                                "intensity": inten_vals.tolist(),
                                                "h5_name": h5_name,
                                            }
                                        )
                                    else:
                                        return_list.append(
                                            {
                                                "spec_name": spec_name,
                                                "smiles": smi,
                                                "mz": [],
                                                "intensity": [],
                                                "h5_name": h5_name,
                                            }
                                        )
                                else:
                                    return_list.append(
                                        {"spec_name": spec_name, "smiles": smi, "spec": output_spec.tolist(), "h5_name": h5_name}
                                    )
                        else:
                            for output_spec, output_frag, spec_name, smi, h5_name in zip(
                                single_outputs["spec"],
                                single_outputs["frag"],
                                [spec_names[i]],
                                [smis[i]],
                                [h5_names[i]],
                            ):
                                if kwargs["sparse_out"]:
                                    threshold = kwargs.get("threshold", 0.01)
                                    valid_inds = np.where(output_spec[:, 1] > threshold)[0]
                                    if len(valid_inds) > 0:
                                        if kwargs["sparse_k"] is not None:
                                            top_k = min(kwargs["sparse_k"], len(valid_inds))
                                            top_k_inds = np.argsort(output_spec[valid_inds, 1])[-top_k:][::-1]
                                            valid_inds = valid_inds[top_k_inds]
                                        mz_vals = output_spec[valid_inds, 0]
                                        inten_vals = output_spec[valid_inds, 1]
                                        frag_vals = [output_frag[j] for j in valid_inds]
                                        return_list.append(
                                            {
                                                "spec_name": spec_name,
                                                "smiles": smi,
                                                "mz": mz_vals.tolist(),
                                                "intensity": inten_vals.tolist(),
                                                "frag": frag_vals,
                                                "h5_name": h5_name,
                                            }
                                        )
                                    else:
                                        return_list.append(
                                            {
                                                "spec_name": spec_name,
                                                "smiles": smi,
                                                "mz": [],
                                                "intensity": [],
                                                "frag": [],
                                                "h5_name": h5_name,
                                            }
                                        )
                                else:
                                    return_list.append(
                                        {
                                            "spec_name": spec_name,
                                            "smiles": smi,
                                            "spec": output_spec.tolist(),
                                            "frag": output_frag,
                                            "h5_name": h5_name,
                                        }
                                    )
                    except Exception as ei:
                        # Record the bad entry, then continue.
                        try:
                            inchikey = common.inchikey_from_smiles(smis[i])
                        except Exception:
                            inchikey = ""

                        skipped_records.append(
                            {
                                "spec": spec_names[i],
                                "smiles": smis[i],
                                "inchikey": inchikey,
                                "collision_energy": colli_eng_vals[i],
                                "ionization": adducts[i],
                                "instrument": instruments[i],
                                "precursor": precursor_mzs[i],
                                "error_type": type(ei).__name__,
                                "error_message": str(ei),
                            }
                        )
                return return_list

            return_list = []
            if binned_out:
                for output_spec, spec_name, smi, h5_name in \
                        zip(full_outputs["spec"], spec_names, smis, h5_names):
                    if kwargs["sparse_out"]:
                        threshold = kwargs.get("threshold", 0.01)
                        valid_inds = np.where(output_spec > threshold)[0]
                        if len(valid_inds) > 0:
                            if kwargs["sparse_k"] is not None:
                                top_k = min(kwargs["sparse_k"], len(valid_inds))
                                top_k_inds = np.argsort(output_spec[valid_inds])[-top_k:][::-1]
                                valid_inds = valid_inds[top_k_inds]
                            mz_vals = valid_inds * common.BIN_SIZE + common.MIN_MASS
                            inten_vals = output_spec[valid_inds]
                            return_list.append({
                                "spec_name": spec_name,
                                "smiles": smi,
                                "mz": mz_vals.tolist(),
                                "intensity": inten_vals.tolist(),
                                "h5_name": h5_name,
                            })
                        else:
                            return_list.append({
                                "spec_name": spec_name,
                                "smiles": smi,
                                "mz": [],
                                "intensity": [],
                                "h5_name": h5_name,
                            })
                    else:
                        return_list.append({
                            "spec_name": spec_name,
                            "smiles": smi,
                            "spec": output_spec.tolist(),
                            "h5_name": h5_name,
                        })
            else:
                for output_spec, output_frag, spec_name, smi, h5_name in \
                        zip(full_outputs["spec"], full_outputs["frag"], spec_names, smis, h5_names):
                    if kwargs["sparse_out"]:
                        threshold = kwargs.get("threshold", 0.01)
                        valid_inds = np.where(output_spec[:, 1] > threshold)[0]
                        if len(valid_inds) > 0:
                            if kwargs["sparse_k"] is not None:
                                top_k = min(kwargs["sparse_k"], len(valid_inds))
                                top_k_inds = np.argsort(output_spec[valid_inds, 1])[-top_k:][::-1]
                                valid_inds = valid_inds[top_k_inds]
                            mz_vals = output_spec[valid_inds, 0]
                            inten_vals = output_spec[valid_inds, 1]
                            frag_vals = [output_frag[i] for i in valid_inds]
                            return_list.append({
                                "spec_name": spec_name,
                                "smiles": smi,
                                "mz": mz_vals.tolist(),
                                "intensity": inten_vals.tolist(),
                                "frag": frag_vals,
                                "h5_name": h5_name,
                            })
                        else:
                            return_list.append({
                                "spec_name": spec_name,
                                "smiles": smi,
                                "mz": [],
                                "intensity": [],
                                "frag": [],
                                "h5_name": h5_name,
                            })
                    else:
                        return_list.append({
                            "spec_name": spec_name,
                            "smiles": smi,
                            "spec": output_spec.tolist(),
                            "frag": output_frag,
                            "h5_name": h5_name,
                        })
            return return_list

        if kwargs["num_workers"] == 0:
            output_entries = [producer_func(batch) for batch in tqdm(all_batched_entries)]
        else:
            output_entries = common.chunked_parallel(
                all_batched_entries,
                producer_func,
                max_cpu=kwargs["num_workers"],
                chunks=10,
            )

        output_entries = [i for j in output_entries for i in j]  # unroll

        # Write output
        logging.info("Writing output")
        import h5py
        output_file = save_dir / "preds.hdf5"
        with h5py.File(output_file, "w") as h5_file:
            for entry in tqdm(output_entries):
                h5_name = entry["h5_name"]
                if binned_out:
                    if kwargs["sparse_out"]:
                        h5_file.create_dataset(
                            f"{h5_name}/spec",
                            data=np.array([entry["mz"], entry["intensity"]]).T,
                        )
                    else:
                        h5_file.create_dataset(
                            f"{h5_name}/spec",
                            data=np.array(entry["spec"]),
                        )
                else:
                    if kwargs["sparse_out"]:
                        h5_file.create_dataset(
                            f"{h5_name}/spec",
                            data=np.array([entry["mz"], entry["intensity"]]).T,
                        )
                        h5_file.create_dataset(
                            f"{h5_name}/frag",
                            data=json.dumps(entry["frag"]).encode("utf-8"),
                        )
                    else:
                        h5_file.create_dataset(
                            f"{h5_name}/spec",
                            data=np.array(entry["spec"]),
                        )
                        h5_file.create_dataset(
                            f"{h5_name}/frag",
                            data=json.dumps(entry["frag"]).encode("utf-8"),
                        )

        logging.info(f"Saved predictions to {output_file}")

        # Save skipped bad molecules info (main-process only; wrapper sets num-workers=0).
        if skip_bad_molecules and skipped_records:
            skipped_csv_path = save_dir / skipped_csv_name
            pd.DataFrame(skipped_records).to_csv(skipped_csv_path, index=False)
            logging.warning(f"Skipped {len(skipped_records)} bad entries. CSV saved to: {skipped_csv_path}")
        elif skip_bad_molecules:
            logging.info("No bad entries were skipped.")


if __name__ == "__main__":
    predict()
