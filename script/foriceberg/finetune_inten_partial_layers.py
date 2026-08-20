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

Fine-tune ``IntenGNN`` from a checkpoint while training **only selected modules**.

Top-level ``IntenGNN`` blocks (use with ``--train-modules`` or ``--preset``):

  gnn              Fragment / molecule GGNN
  root_module      Root encoding (often same weights as ``gnn``)
  trans_layers     Transformer stack **before** the MLP head
  intermediate_out MLP between pooling and spectrum linear head
  output_map       Final linear → binned intensities
  isomer_attn_out  Isomer-attention linear (parallel to ``output_map``)
  embedder         Formula embedding MLP (if ``encode_forms`` was enabled)

Presets (``--preset``):

  head             ``output_map``, ``isomer_attn_out``
  mlp_head         ``intermediate_out``, ``output_map``, ``isomer_attn_out``
  before_mlp       ``trans_layers`` only (Transformer before MLP)
  transformer      alias of ``before_mlp``
  gnn              ``gnn``, ``root_module``
  all              all parameters (full fine-tune)

Example::

    cd /path/to/FMS2Lib
    python3 script/foriceberg/finetune_inten_partial_layers.py \
      --pretrained-checkpoint mode/foriceberg/intensity/best.ckpt \
      --save-dir results/inten_finetune_before_mlp \
      --preset before_mlp \
      --dataset-name F_ICEBERG_dataset \
      --split-name split_1.tsv \
      --magma-dag-folder results/.../preds_train_100_inten.hdf5 \
      --gpu
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import types
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader

torch.multiprocessing.set_sharing_strategy("file_system")

# --- FMS2Lib layout ------------------------------------------------------
_SCRIPT = Path(__file__).resolve()
_SCRIPT_DIR = _SCRIPT.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from foriceberg_paths import repo_root, pythonpath_src  # noqa: E402

_REPO_ROOT = repo_root()
if str(pythonpath_src()) not in sys.path:
    sys.path.insert(0, pythonpath_src())

import ms_pred.common as common  # noqa: E402
from ms_pred.dag_pred import dag_data, inten_model  # noqa: E402
from ms_pred.nn_utils import nn_utils  # noqa: E402

# ---------------------------------------------------------------------------

PRESETS: dict[str, list[str]] = {
    "head": ["output_map", "isomer_attn_out"],
    "mlp_head": ["intermediate_out", "output_map", "isomer_attn_out"],
    "before_mlp": ["trans_layers"],
    "transformer": ["trans_layers"],
    "gnn": ["gnn", "root_module"],
    "all": [],  # special: train everything
}

TOP_LEVEL_HINTS = sorted(
    {
        "gnn",
        "root_module",
        "trans_layers",
        "intermediate_out",
        "output_map",
        "isomer_attn_out",
        "embedder",
        "pool",
    }
)


def _parse_modules(raw: str | None) -> list[str]:
    if not raw or not raw.strip():
        return []
    return [m.strip() for m in raw.replace(";", ",").split(",") if m.strip()]


def resolve_train_modules(preset: str | None, train_modules: list[str]) -> list[str]:
    if preset:
        key = preset.strip().lower()
        if key not in PRESETS:
            raise ValueError(f"Unknown preset {preset!r}. Choose from: {', '.join(sorted(PRESETS))}")
        if key == "all":
            return []
        preset_mods = PRESETS[key]
        extra = [m for m in train_modules if m not in preset_mods]
        return preset_mods + extra
    if not train_modules:
        raise ValueError("Specify --train-modules and/or --preset (e.g. --preset before_mlp)")
    return train_modules


def apply_partial_training(model: torch.nn.Module, modules: list[str]) -> dict[str, int]:
    """Freeze all params, then enable gradients for matching name prefixes."""
    for p in model.parameters():
        p.requires_grad = False

    if not modules:
        for p in model.parameters():
            p.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        return {"trainable": trainable, "total": total, "matched_prefixes": []}

    matched_prefixes: set[str] = set()
    for name, param in model.named_parameters():
        for pref in modules:
            if name == pref or name.startswith(pref + "."):
                param.requires_grad = True
                matched_prefixes.add(pref)
                break

    missing = set(modules) - matched_prefixes
    if missing:
        logging.warning(
            "No parameters matched these module prefixes (check spelling): %s",
            sorted(missing),
        )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if trainable == 0:
        raise RuntimeError(
            f"No trainable parameters after applying modules={modules}. "
            f"Run with --list-modules to inspect names."
        )
    return {
        "trainable": trainable,
        "total": total,
        "matched_prefixes": sorted(matched_prefixes),
    }


def patch_optimizer_for_trainable_only(model: inten_model.IntenGNN) -> None:
    """``configure_optimizers`` only passes parameters with ``requires_grad=True``."""

    def _configure_optimizers(self):  # type: ignore[no-untyped-def]
        trainable = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(
            trainable, lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = nn_utils.build_lr_scheduler(
            optimizer=optimizer, lr_decay_rate=self.lr_decay_rate, warmup=self.warmup
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "frequency": 1,
                "interval": "step",
            },
        }

    model.configure_optimizers = types.MethodType(_configure_optimizers, model)


def print_module_summary(model: torch.nn.Module) -> None:
    print("Top-level child modules on IntenGNN:")
    for name in TOP_LEVEL_HINTS:
        if hasattr(model, name):
            sub = getattr(model, name)
            n = sum(p.numel() for p in sub.parameters())
            print(f"  {name}: {n:,} params")
    print("\nParameter prefixes (first 40 lines):")
    for i, (name, p) in enumerate(model.named_parameters()):
        if i >= 40:
            print("  ...")
            break
        print(f"  {name}  requires_grad={p.requires_grad}")


def build_magma_map(magma_dag_folder: Path) -> dict[str, str]:
    magma_tree_h5 = common.HDF5Dataset(magma_dag_folder)
    name_to_json: dict[str, str] = {}
    for i in magma_tree_h5.get_all_names():
        stem = Path(i).stem
        if stem.endswith("_spec"):
            base_name = stem[:-5]
        elif "_collision" in stem:
            base_name = stem.split("_collision")[0]
        else:
            base_name = stem.replace("pred_", "")
        name_to_json[base_name] = i
    return name_to_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Partial-layer fine-tune for IntenGNN (freeze all but selected modules)."
    )
    parser.add_argument("--pretrained-checkpoint", required=True, type=str)
    parser.add_argument("--save-dir", required=True, type=str)
    parser.add_argument(
        "--preset",
        default=None,
        choices=sorted(PRESETS.keys()),
        help="Shortcut module group (combine with --train-modules to add more)",
    )
    parser.add_argument(
        "--train-modules",
        default="",
        type=str,
        help="Comma-separated module prefixes, e.g. trans_layers,intermediate_out",
    )
    parser.add_argument("--list-modules", action="store_true", help="Print module tree and exit")
    parser.add_argument("--dataset-name", default="F_ICEBERG_dataset", type=str)
    parser.add_argument("--dataset-labels", default="labels.tsv", type=str)
    parser.add_argument("--split-name", default="split_1.tsv", type=str)
    parser.add_argument("--magma-dag-folder", required=False, type=str)
    parser.add_argument("--learning-rate", default=1e-4, type=float)
    parser.add_argument("--lr-decay-rate", default=0.825, type=float)
    parser.add_argument("--weight-decay", default=1e-7, type=float)
    parser.add_argument("--max-epochs", default=50, type=int)
    parser.add_argument("--min-epochs", default=0, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--num-workers", default=8, type=int)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--seed", default=1, type=int)
    args = parser.parse_args()

    os.chdir(_REPO_ROOT)
    ckpt = Path(args.pretrained_checkpoint)
    if not ckpt.is_file() and not (_REPO_ROOT / ckpt).is_file():
        print(f"ERROR: checkpoint not found: {ckpt}", file=sys.stderr)
        sys.exit(2)
    ckpt_path = ckpt if ckpt.is_file() else _REPO_ROOT / ckpt

    model = inten_model.IntenGNN.load_from_checkpoint(str(ckpt_path))

    if args.list_modules:
        print_module_summary(model)
        return

    if not args.magma_dag_folder:
        parser.error("--magma-dag-folder is required unless --list-modules")

    modules = resolve_train_modules(args.preset, _parse_modules(args.train_modules))
    stats = apply_partial_training(model, modules)
    patch_optimizer_for_trainable_only(model)

    model.learning_rate = args.learning_rate
    model.lr_decay_rate = args.lr_decay_rate
    model.weight_decay = args.weight_decay

    save_dir = Path(args.save_dir)
    if not save_dir.is_absolute():
        save_dir = _REPO_ROOT / save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    common.setup_logger(save_dir, log_name="dag_inten_partial_finetune.log", debug=False)
    pl.seed_everything(args.seed)

    run_meta = {**vars(args), "train_modules_resolved": modules, **stats}
    with open(save_dir / "args.yaml", "w") as fp:
        fp.write(yaml.dump(run_meta, default_flow_style=False))

    logging.info("Checkpoint: %s", ckpt_path)
    logging.info("Train modules: %s", modules or ["ALL"])
    logging.info(
        "Trainable params: %s / %s (%.2f%%)",
        f"{stats['trainable']:,}",
        f"{stats['total']:,}",
        100.0 * stats["trainable"] / max(stats["total"], 1),
    )

    data_dir = common.get_data_dir(args.dataset_name)
    labels = data_dir / args.dataset_labels
    split_file = data_dir / "splits" / args.split_name
    df = pd.read_csv(labels, sep="\t")
    spec_names = df["spec"].values
    train_inds, val_inds, test_inds = common.get_splits(spec_names, str(split_file))
    train_df, val_df, test_df = df.iloc[train_inds], df.iloc[val_inds], df.iloc[test_inds]

    magma_dag_folder = Path(args.magma_dag_folder)
    if not magma_dag_folder.is_absolute():
        magma_dag_folder = _REPO_ROOT / magma_dag_folder
    name_to_json = build_magma_map(magma_dag_folder)

    tree_processor = dag_data.TreeProcessor(
        pe_embed_k=model.pe_embed_k,
        root_encode=model.root_encode,
        binned_targs=model.binned_targs,
        add_hs=model.add_hs,
        embed_elem_group=model.embed_elem_group,
    )
    train_dataset = dag_data.IntenDataset(
        train_df,
        magma_h5=magma_dag_folder,
        magma_map=name_to_json,
        num_workers=args.num_workers,
        tree_processor=tree_processor,
    )
    val_dataset = dag_data.IntenDataset(
        val_df,
        magma_h5=magma_dag_folder,
        magma_map=name_to_json,
        num_workers=args.num_workers,
        tree_processor=tree_processor,
    )
    collate_fn = train_dataset.get_collate_fn()
    train_loader = DataLoader(
        train_dataset,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        shuffle=True,
        batch_size=args.batch_size,
        pin_memory=args.gpu,
    )
    val_loader = DataLoader(
        val_dataset,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        shuffle=False,
        batch_size=args.batch_size,
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
        EarlyStopping(monitor="val_loss", patience=5),
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

    logging.info("Starting partial fine-tune...")
    trainer.fit(model, train_loader, val_loader)
    logging.info("Best checkpoint: %s", checkpoint_callback.best_model_path)

    test_dataset = dag_data.IntenDataset(
        test_df,
        magma_h5=magma_dag_folder,
        magma_map=name_to_json,
        num_workers=args.num_workers,
        tree_processor=tree_processor,
    )
    test_loader = DataLoader(
        test_dataset,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        shuffle=False,
        batch_size=args.batch_size,
        pin_memory=args.gpu,
    )
    best = inten_model.IntenGNN.load_from_checkpoint(checkpoint_callback.best_model_path)
    best.eval()
    trainer.test(model=best, dataloaders=test_loader)


if __name__ == "__main__":
    main()
