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

Full fine-tune of ``FragGNN`` (generation model): load a pretrained checkpoint and
train **all** parameters (no freezing).

Requires MAGMA trees at
``data/foriceberg/spec_datasets/<dataset>/magma_outputs/magma_tree.hdf5``.

Example::

    cd /path/to/FMS2Lib
    python3 script/foriceberg/finetune_gen_full.py \
      --pretrained-checkpoint mode/foriceberg/generation/best.ckpt \
      --dataset-name F_ICEBERG_dataset \
      --split-name split_1.tsv \
      --save-dir script/foriceberg/results/gen_finetune_full \
      --learning-rate 1e-4 \
      --max-epochs 200 \
      --gpu
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader

torch.multiprocessing.set_sharing_strategy("file_system")

_SCRIPT = Path(__file__).resolve()
_SCRIPT_DIR = _SCRIPT.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from foriceberg_paths import repo_root, pythonpath_src  # noqa: E402

_REPO_ROOT = repo_root()
if str(pythonpath_src()) not in sys.path:
    sys.path.insert(0, pythonpath_src())

import ms_pred.common as common  # noqa: E402
from ms_pred.dag_pred import dag_data, gen_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full fine-tune FragGNN from a pretrained checkpoint (no freezing)."
    )
    parser.add_argument("--pretrained-checkpoint", required=True, type=str)
    parser.add_argument("--save-dir", required=True, type=str)
    parser.add_argument("--dataset-name", required=True, type=str)
    parser.add_argument("--dataset-labels", default="labels.tsv", type=str)
    parser.add_argument("--split-name", default="split_1.tsv", type=str)
    parser.add_argument("--magma-folder", default="magma_outputs", type=str)
    parser.add_argument("--learning-rate", default=1e-4, type=float)
    parser.add_argument("--lr-decay-rate", default=0.825, type=float)
    parser.add_argument("--weight-decay", default=0.0, type=float)
    parser.add_argument("--max-epochs", default=200, type=int)
    parser.add_argument("--min-epochs", default=0, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--num-workers", default=8, type=int)
    parser.add_argument("--patience", default=5, type=int)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--seed", default=1, type=int)
    args = parser.parse_args()

    os.chdir(_REPO_ROOT)
    ckpt = Path(args.pretrained_checkpoint)
    ckpt_path = ckpt if ckpt.is_file() else _REPO_ROOT / ckpt
    if not ckpt_path.is_file():
        print(f"ERROR: checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(2)

    model = gen_model.FragGNN.load_from_checkpoint(str(ckpt_path))
    for p in model.parameters():
        p.requires_grad = True

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())

    model.learning_rate = args.learning_rate
    model.lr_decay_rate = args.lr_decay_rate
    model.weight_decay = args.weight_decay

    save_dir = Path(args.save_dir)
    if not save_dir.is_absolute():
        save_dir = _REPO_ROOT / save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    common.setup_logger(save_dir, log_name="dag_gen_full_finetune.log", debug=False)
    pl.seed_everything(args.seed)

    with open(save_dir / "args.yaml", "w") as fp:
        fp.write(
            yaml.dump(
                {
                    **vars(args),
                    "pretrained_checkpoint_resolved": str(ckpt_path),
                    "trainable_params": n_train,
                    "total_params": n_total,
                    "freeze": False,
                },
                default_flow_style=False,
            )
        )

    logging.info("Checkpoint: %s", ckpt_path)
    logging.info("Trainable params: %s / %s (all layers)", f"{n_train:,}", f"{n_total:,}")

    data_dir = common.get_data_dir(args.dataset_name)
    labels = data_dir / args.dataset_labels
    split_file = data_dir / "splits" / args.split_name
    magma_h5 = data_dir / args.magma_folder / "magma_tree.hdf5"
    if not magma_h5.is_file():
        print(f"ERROR: missing MAGMA trees: {magma_h5}", file=sys.stderr)
        sys.exit(2)

    df = pd.read_csv(labels, sep="\t")
    train_inds, val_inds, test_inds = common.get_splits(df["spec"].values, str(split_file))
    train_df, val_df, test_df = df.iloc[train_inds], df.iloc[val_inds], df.iloc[test_inds]
    logging.info("Split sizes: train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))

    magma_tree = common.HDF5Dataset(magma_h5)
    name_to_json = {Path(i).stem: i for i in magma_tree.get_all_names()}
    magma_tree.close()

    tree_processor = dag_data.TreeProcessor(
        pe_embed_k=model.pe_embed_k,
        root_encode=model.root_encode,
        add_hs=model.add_hs,
        embed_elem_group=model.embed_elem_group,
    )
    train_dataset = dag_data.GenDataset(
        train_df,
        magma_h5=magma_h5,
        magma_map=name_to_json,
        num_workers=args.num_workers,
        tree_processor=tree_processor,
    )
    val_dataset = dag_data.GenDataset(
        val_df,
        magma_h5=magma_h5,
        magma_map=name_to_json,
        num_workers=args.num_workers,
        tree_processor=tree_processor,
    )
    test_dataset = dag_data.GenDataset(
        test_df,
        magma_h5=magma_h5,
        magma_map=name_to_json,
        num_workers=args.num_workers,
        tree_processor=tree_processor,
    )

    collate_fn = train_dataset.get_collate_fn()
    persistent_workers = args.num_workers > 0
    mp_context = "spawn" if args.num_workers > 0 else None
    train_loader = DataLoader(
        train_dataset,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        shuffle=True,
        batch_size=args.batch_size,
        persistent_workers=persistent_workers,
        multiprocessing_context=mp_context,
        pin_memory=args.gpu,
    )
    val_loader = DataLoader(
        val_dataset,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        shuffle=False,
        batch_size=args.batch_size,
        persistent_workers=persistent_workers,
        multiprocessing_context=mp_context,
        pin_memory=args.gpu,
    )
    test_loader = DataLoader(
        test_dataset,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        shuffle=False,
        batch_size=args.batch_size,
        persistent_workers=persistent_workers,
        multiprocessing_context=mp_context,
        pin_memory=args.gpu,
    )

    tb_logger = pl_loggers.TensorBoardLogger(save_dir, name="")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=tb_logger.log_dir,
        filename="best",
        save_weights_only=False,
    )
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=args.patience),
        checkpoint_callback,
        LearningRateMonitor(logging_interval="epoch"),
    ]
    trainer = pl.Trainer(
        logger=[tb_logger, common.ConsoleLogger()],
        accelerator="gpu" if args.gpu else "cpu",
        devices=1 if args.gpu else "auto",
        callbacks=callbacks,
        gradient_clip_val=5,
        gradient_clip_algorithm="value",
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        num_sanity_val_steps=0,
    )

    logging.info("Starting full gen fine-tune (no freeze)...")
    trainer.fit(model, train_loader, val_loader)
    logging.info("Best checkpoint: %s", checkpoint_callback.best_model_path)

    if checkpoint_callback.best_model_path:
        best = gen_model.FragGNN.load_from_checkpoint(checkpoint_callback.best_model_path)
        best.eval()
        trainer.test(model=best, dataloaders=test_loader)


if __name__ == "__main__":
    main()
