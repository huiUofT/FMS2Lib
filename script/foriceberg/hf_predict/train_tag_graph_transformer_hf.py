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

Author: Tao Yang
Date: 20260819

This code was independently developed by the author.
All data-processing, modeling, and evaluation procedures were implemented
specifically for this study.
"""

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score, average_precision_score, classification_report
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch
from torch_geometric.data import Data as PyGData

from smiles_graph_transformer_hf import SmilesGraphTransformer

COLS = ["HF"]
COL2ID = {c: i for i, c in enumerate(COLS)}
RWSE_STEPS = 8


def compute_rwse(mol: Chem.Mol, num_steps: int = RWSE_STEPS) -> torch.Tensor:
    """Return node-wise random-walk return probabilities for k=1..num_steps."""
    n = mol.GetNumAtoms()
    a = torch.zeros((n, n), dtype=torch.float32)
    for i in range(n):
        a[i, i] = 1.0  # self-loop
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        a[i, j] = 1.0
        a[j, i] = 1.0

    deg = a.sum(dim=1, keepdim=True).clamp_min(1.0)
    p = a / deg

    rwse_cols = []
    pk = p.clone()
    for _ in range(num_steps):
        rwse_cols.append(torch.diagonal(pk, 0))
        pk = pk @ p
    return torch.stack(rwse_cols, dim=1)


def one_hot(x, choices):
    return [1 if x == c else 0 for c in choices]


def atom_features(a: Chem.Atom) -> List[float]:
    # Node features for HF prediction.
    # Keep the feature definition consistent with the corresponding inference script.
    elem_choices = ["Br", "C", "Cl", "F", "N", "O", "S"]  # 7-dim one-hot
    hyb_sp_choices = [  # hybridization: sp/sp2/sp3/other
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
    ]

    feats: List[float] = []
    feats += one_hot(a.GetSymbol(), elem_choices)

    # 1) is_aromatic (0/1)
    feats.append(1.0 if a.GetIsAromatic() else 0.0)

    # 2) hybridization one-hot: sp, sp2, sp3, other (4-dim)
    hyb = a.GetHybridization()
    is_sp = 1.0 if hyb == hyb_sp_choices[0] else 0.0
    is_sp2 = 1.0 if hyb == hyb_sp_choices[1] else 0.0
    is_sp3 = 1.0 if hyb == hyb_sp_choices[2] else 0.0
    is_other = 1.0 if (hyb != hyb_sp_choices[0] and hyb != hyb_sp_choices[1] and hyb != hyb_sp_choices[2]) else 0.0
    feats += [is_sp, is_sp2, is_sp3, is_other]

    # 3) is_in_ring (0/1)
    feats.append(1.0 if a.IsInRing() else 0.0)

    # 4) total H count one-hot for 0..6 (clip >6 into 6)
    hcount = int(a.GetTotalNumHs())
    hcount = min(max(hcount, 0), 6)
    feats += one_hot(hcount, list(range(7)))

    # 5) carbon directly bonded to fluorine: C-F adjacency (1 if atom is carbon and has F neighbor)
    connected_to_f = 1.0 if (a.GetSymbol() == "C" and any(n.GetSymbol() == "F" for n in a.GetNeighbors())) else 0.0
    feats.append(connected_to_f)

    return feats


def bond_features(b: Chem.Bond) -> List[float]:
    feats = []
    feats += one_hot(str(b.GetBondType()), ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"])
    feats.append(1 if b.GetIsConjugated() else 0)
    feats.append(1 if b.IsInRing() else 0)
    return feats


def mol_to_pyg_from_smiles(smiles: str) -> PyGData:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.RemoveHs(mol)
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float32)

    edge_index = []
    edge_attr = []
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        bf = bond_features(b)
        edge_index.append([i, j])
        edge_attr.append(bf)
        edge_index.append([j, i])
        edge_attr.append(bf)

    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, len(bond_features(Chem.MolFromSmiles("CC").GetBondWithIdx(0)))), dtype=torch.float32)

    rwse = compute_rwse(mol, num_steps=RWSE_STEPS)
    return PyGData(x=x, edge_index=edge_index, edge_attr=edge_attr, rwse=rwse)


def build_graphs_from_df(df: pd.DataFrame) -> Dict[str, PyGData]:
    graphs: Dict[str, PyGData] = {}
    for mol_id, smiles in df[["mol_id", "smiles"]].drop_duplicates().itertuples(index=False):
        try:
            graphs[str(mol_id)] = mol_to_pyg_from_smiles(str(smiles))
        except Exception:
            continue
    return graphs


def build_scaffold_groups(df: pd.DataFrame, include_chirality: bool = False) -> np.ndarray:
    groups = []
    for smiles, mol_id in df[["smiles", "mol_id"]].itertuples(index=False):
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            groups.append(f"INVALID::{mol_id}")
            continue
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        if scaf is None or scaf.GetNumAtoms() == 0:
            groups.append(f"NO_SCAF::{mol_id}")
            continue
        groups.append(Chem.MolToSmiles(scaf, isomericSmiles=bool(include_chirality)))
    return np.asarray(groups, dtype=object)


class SmilesDataset(Dataset):
    def __init__(self, df: pd.DataFrame, graphs: Dict[str, PyGData]):
        self.df = df.reset_index(drop=True).copy()
        self.graphs = graphs
        keep = self.df["mol_id"].astype(str).isin(set(graphs.keys()))
        self.df = self.df.loc[keep].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mol_id = str(row["mol_id"])
        graph = self.graphs[mol_id]
        col_id = torch.tensor(COL2ID["HF"], dtype=torch.long)
        y = int(row["class_label"])
        return mol_id, graph, col_id, y


def collate_fn(batch):
    mol_ids, graphs, col_ids, ys = zip(*batch)
    batch_graph = Batch.from_data_list(list(graphs))
    col_ids = torch.stack(col_ids, dim=0)
    ys = torch.tensor(ys, dtype=torch.long)
    return mol_ids, batch_graph, col_ids, ys


def compute_class_weights(y: np.ndarray, n_classes: int = 2) -> torch.Tensor:
    counts = np.bincount(y.astype(int), minlength=n_classes)
    weights = np.zeros_like(counts, dtype=np.float32)
    nz = counts > 0
    weights[nz] = counts.sum() / (len(counts[nz]) * counts[nz])
    return torch.tensor(weights, dtype=torch.float32)


def summarize_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, default="data/foriceberg/hf_predict/tag.csv")
    ap.add_argument("--out_dir", type=str, default="script/foriceberg/results/hf_predict/output")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--hidden_dim", type=int, default=192)
    ap.add_argument("--fusion_hidden", type=int, default=256)
    ap.add_argument("--num_layers", type=int, default=5)
    ap.add_argument("--ggnn_layers", type=int, default=3)
    ap.add_argument("--transformer_layers", type=int, default=2)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--col_emb_dim", type=int, default=24)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--test_frac", type=float, default=0.1)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--split_mode", type=str, default="scaffold", choices=["scaffold", "group_mol"])
    ap.add_argument("--scaffold_include_chirality", action="store_true")
    ap.add_argument("--use_class_weights", dest="use_class_weights", action="store_true")
    ap.add_argument("--no_class_weights", dest="use_class_weights", action="store_false")
    ap.set_defaults(use_class_weights=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.train_csv)
    required = {"FEATURE_ID", "SMILES", "produces_HF"}
    if not required.issubset(raw.columns):
        raise ValueError(f"train_csv must contain columns: {sorted(required)}")

    df = pd.DataFrame({
        "mol_id": raw["FEATURE_ID"].astype(str),
        "smiles": raw["SMILES"].astype(str),
        "class_label": raw["produces_HF"].astype(str).str.upper().map({"FALSE": 0, "TRUE": 1}),
    })
    df = df.dropna(subset=["smiles", "class_label"]).copy()
    df["class_label"] = df["class_label"].astype(int)

    graphs = build_graphs_from_df(df)
    if len(graphs) == 0:
        raise RuntimeError("No valid SMILES graphs could be built.")

    any_mid = next(iter(graphs))
    g0 = graphs[any_mid]
    node_in_dim = g0.x.shape[1]
    edge_dim = g0.edge_attr.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("node_in_dim:", node_in_dim, "edge_dim:", edge_dim)

    if args.split_mode == "group_mol":
        groups = df["mol_id"].astype(str).to_numpy()
        print(f"Split mode: group_mol, groups={len(set(groups.tolist()))}")
    else:
        groups = build_scaffold_groups(df, include_chirality=bool(args.scaffold_include_chirality))
        print(f"Split mode: scaffold, groups={len(set(groups.tolist()))}, include_chirality={bool(args.scaffold_include_chirality)}")

    if args.val_frac <= 0 or args.test_frac <= 0 or (args.val_frac + args.test_frac) >= 1:
        raise ValueError("Require 0 < val_frac < 1, 0 < test_frac < 1, and val_frac + test_frac < 1.")

    splitter_test = GroupShuffleSplit(n_splits=1, test_size=float(args.test_frac), random_state=args.seed)
    trva_idx, te_idx = next(splitter_test.split(df, groups=groups))

    rem_df = df.iloc[trva_idx].reset_index(drop=True).copy()
    rem_groups = groups[trva_idx]
    val_over_rem = float(args.val_frac) / (1.0 - float(args.test_frac))
    splitter_val = GroupShuffleSplit(n_splits=1, test_size=val_over_rem, random_state=args.seed + 1)
    tr_rel_idx, va_rel_idx = next(splitter_val.split(rem_df, groups=rem_groups))

    tr_idx = trva_idx[tr_rel_idx]
    va_idx = trva_idx[va_rel_idx]

    tr_df = df.iloc[tr_idx].reset_index(drop=True).copy()
    va_df = df.iloc[va_idx].reset_index(drop=True).copy()
    te_df = df.iloc[te_idx].reset_index(drop=True).copy()

    print(f"Split sizes -> train: {len(tr_df)}, val: {len(va_df)}, test: {len(te_df)}")

    tr_ds = SmilesDataset(tr_df, graphs)
    va_ds = SmilesDataset(va_df, graphs)
    te_ds = SmilesDataset(te_df, graphs)

    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    te_loader = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    model = SmilesGraphTransformer(
        node_in_dim=node_in_dim,
        edge_dim=edge_dim,
        num_columns=len(COLS),
        num_classes=2,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        ggnn_layers=args.ggnn_layers,
        transformer_layers=args.transformer_layers,
        num_heads=args.num_heads,
        col_emb_dim=args.col_emb_dim,
        descriptor_dim=0,
        fusion_hidden=args.fusion_hidden,
        dropout=args.dropout,
        rwse_dim=RWSE_STEPS,
    ).to(device)

    weight = compute_class_weights(tr_df["class_label"].to_numpy(), n_classes=2) if args.use_class_weights else None
    if weight is not None:
        weight = weight.to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_score = -np.inf
    best_path = out_dir / "best_hf_binary.pt"
    best_val_rows = []

    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        for _, batch_graph, col_id, y in tr_loader:
            batch_graph = batch_graph.to(device)
            col_id = col_id.to(device)
            y = y.to(device)

            logits = model(batch_graph, col_id)
            loss = criterion(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu().item()))

        model.eval()
        val_true, val_pred, val_prob1 = [], [], []
        epoch_rows = []
        with torch.no_grad():
            for mol_ids, batch_graph, col_id, y in va_loader:
                batch_graph = batch_graph.to(device)
                col_id = col_id.to(device)
                logits = model(batch_graph, col_id)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                pred = probs.argmax(axis=1)
                yt = y.numpy()
                val_true.extend(yt.tolist())
                val_pred.extend(pred.tolist())
                val_prob1.extend(probs[:, 1].tolist())
                for i, mid in enumerate(mol_ids):
                    epoch_rows.append({
                        "mol_id": mid,
                        "true_class": int(yt[i]),
                        "pred_class": int(pred[i]),
                        "p_class0": float(probs[i, 0]),
                        "p_class1": float(probs[i, 1]),
                    })

        metrics = summarize_metrics(val_true, val_pred)
        val_roc = roc_auc_score(val_true, val_prob1) if len(set(val_true)) > 1 else float("nan")
        val_pr = average_precision_score(val_true, val_prob1) if len(set(val_true)) > 1 else float("nan")
        print(
            f"ep {ep:03d} train_loss={np.mean(losses):.4f} "
            f"acc={metrics['accuracy']:.3f} bal_acc={metrics['balanced_accuracy']:.3f} "
            f"macro_f1={metrics['macro_f1']:.3f} ROC-AUC={val_roc:.3f} PR-AUC={val_pr:.3f}"
        )

        if metrics["macro_f1"] > best_score:
            best_score = metrics["macro_f1"]
            best_val_rows = epoch_rows
            torch.save({
                "model_state": model.state_dict(),
                "columns": COLS,
                "node_in_dim": node_in_dim,
                "edge_dim": edge_dim,
                "num_classes": 2,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "ggnn_layers": args.ggnn_layers,
                "transformer_layers": args.transformer_layers,
                "num_heads": args.num_heads,
                "col_emb_dim": args.col_emb_dim,
                "rwse_dim": RWSE_STEPS,
                "descriptor_dim": 0,
                "fusion_hidden": args.fusion_hidden,
                "dropout": args.dropout,
                "model_type": "smiles_graph_transformer_hf_binary",
                "source_csv": str(args.train_csv),
            }, best_path)

    pd.DataFrame(best_val_rows).to_csv(out_dir / "val_predictions_best_epoch.csv", index=False)
    val_df = pd.DataFrame(best_val_rows)
    if len(val_df) > 0:
        overall = summarize_metrics(val_df["true_class"], val_df["pred_class"])
        overall["confusion_matrix"] = confusion_matrix(val_df["true_class"], val_df["pred_class"], labels=[0, 1]).tolist()
        pd.DataFrame([overall]).to_csv(out_dir / "val_metrics_best_epoch.csv", index=False)

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    test_rows = []
    test_true, test_pred, test_prob1 = [], [], []
    with torch.no_grad():
        for mol_ids, batch_graph, col_id, y in te_loader:
            batch_graph = batch_graph.to(device)
            col_id = col_id.to(device)
            logits = model(batch_graph, col_id)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            pred = probs.argmax(axis=1)
            yt = y.numpy()
            test_true.extend(yt.tolist())
            test_pred.extend(pred.tolist())
            test_prob1.extend(probs[:, 1].tolist())
            for i, mid in enumerate(mol_ids):
                test_rows.append({
                    "mol_id": mid,
                    "true_class": int(yt[i]),
                    "pred_class": int(pred[i]),
                    "p_class0": float(probs[i, 0]),
                    "p_class1": float(probs[i, 1]),
                })

    test_metrics = summarize_metrics(test_true, test_pred)
    test_metrics["roc_auc"] = float(roc_auc_score(test_true, test_prob1)) if len(set(test_true)) > 1 else float("nan")
    test_metrics["pr_auc"] = float(average_precision_score(test_true, test_prob1)) if len(set(test_true)) > 1 else float("nan")
    test_metrics["confusion_matrix"] = confusion_matrix(test_true, test_pred, labels=[0, 1]).tolist()

    pd.DataFrame(test_rows).to_csv(out_dir / "test_predictions_best_model.csv", index=False)
    pd.DataFrame([test_metrics]).to_csv(out_dir / "test_metrics_best_model.csv", index=False)

    print("Saved:", best_path)
    print("Saved:", out_dir / "val_predictions_best_epoch.csv")
    print("Saved:", out_dir / "val_metrics_best_epoch.csv")
    print("Saved:", out_dir / "test_predictions_best_model.csv")
    print("Saved:", out_dir / "test_metrics_best_model.csv")
    print("\n--- Test Classification Report ---")
    print(classification_report(test_true, test_pred, digits=4))


if __name__ == "__main__":
    main()
