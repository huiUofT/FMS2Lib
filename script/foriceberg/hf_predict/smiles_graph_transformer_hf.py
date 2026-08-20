"""
Author: Tao Yang
Date: 20260819

This code was independently developed by the author.
All data-processing, modeling, and evaluation procedures were implemented
specifically for this study.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GatedGraphConv, TransformerConv, global_mean_pool


class SmilesGraphTransformer(nn.Module):
    def __init__(
        self,
        node_in_dim: int,
        edge_dim: int,
        num_columns: int = 1,
        num_classes: int = 2,
        hidden_dim: int = 192,
        num_layers: int = 5,
        ggnn_layers: int = 3,
        transformer_layers: int = 2,
        num_heads: int = 4,
        col_emb_dim: int = 24,
        descriptor_dim: int = 0,
        descriptor_hidden: int = 64,
        fusion_hidden: int = 256,
        dropout: float = 0.1,
        rwse_dim: int = 8,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.dropout = float(dropout)
        self.node_proj = nn.Linear(node_in_dim, hidden_dim)
        self.rwse_dim = int(rwse_dim)
        self.rwse_proj = nn.Linear(self.rwse_dim, hidden_dim)

        # Keep backward compatibility: if explicit split is not provided, fall back to old num_layers.
        if ggnn_layers < 0 or transformer_layers < 0:
            raise ValueError("ggnn_layers and transformer_layers must be non-negative")
        total_layers = int(ggnn_layers) + int(transformer_layers)
        if total_layers <= 0:
            ggnn_layers = int(num_layers)
            transformer_layers = 0
        self.ggnn_layers = int(ggnn_layers)
        self.transformer_layers = int(transformer_layers)

        self.ggnn_convs = nn.ModuleList([
            GatedGraphConv(out_channels=hidden_dim, num_layers=1)
            for _ in range(self.ggnn_layers)
        ])
        self.ggnn_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(self.ggnn_layers)])

        self.trans_convs = nn.ModuleList([
            TransformerConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // num_heads,
                heads=num_heads,
                dropout=dropout,
                edge_dim=edge_dim,
                beta=True,
            )
            for _ in range(self.transformer_layers)
        ])
        self.trans_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(self.transformer_layers)])
        self.site_head = nn.Linear(hidden_dim, 1)
        self.null_site_emb = nn.Parameter(torch.zeros(hidden_dim))
        self.col_emb = nn.Embedding(num_columns, col_emb_dim)
        self.descriptor_dim = int(descriptor_dim)

        if self.descriptor_dim > 0:
            self.descriptor_mlp = nn.Sequential(
                nn.Linear(self.descriptor_dim, descriptor_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(descriptor_hidden, descriptor_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            # graph embedding is concatenation of:
            # - global mean pooling over all nodes
            # - top-scored candidate site embedding (connected_to_f atoms only)
            fusion_in_dim = (2 * hidden_dim) + col_emb_dim + descriptor_hidden
        else:
            self.descriptor_mlp = None
            fusion_in_dim = (2 * hidden_dim) + col_emb_dim

        self.fusion = nn.Sequential(
            nn.Linear(fusion_in_dim, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(fusion_hidden, num_classes)

    def encode_graph(self, batch_graph):
        # Pool only over atoms marked by connected_to_f (last feature dim).
        # connected_to_f is a binary indicator in the node feature vector.
        x_in = batch_graph.x
        connected_to_f_mask = x_in[:, -1] > 0.5

        rwse = getattr(batch_graph, "rwse", None)
        if rwse is None:
            # Backward-compatible fallback for old graphs without RWSE.
            rwse = torch.zeros((x_in.size(0), self.rwse_dim), dtype=x_in.dtype, device=x_in.device)
        else:
            rwse = rwse.to(x_in.device).to(x_in.dtype)

        x = self.node_proj(x_in) + self.rwse_proj(rwse)
        for conv, norm in zip(self.ggnn_convs, self.ggnn_norms):
            h = conv(x, batch_graph.edge_index)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = norm(x + h)

        for conv, norm in zip(self.trans_convs, self.trans_norms):
            h = conv(x, batch_graph.edge_index, batch_graph.edge_attr)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = norm(x + h)

        # Global mean-pool across all nodes (original global pooling).
        g_global = global_mean_pool(x, batch_graph.batch)

        # Score candidate sites (connected_to_f atoms), then pick top-1 site per molecule.
        num_graphs = int(batch_graph.batch.max().item()) + 1 if batch_graph.batch.numel() > 0 else 0
        g_site = x.new_zeros((num_graphs, x.size(1)))
        if num_graphs > 0:
            site_scores = self.site_head(x).squeeze(-1)
            site_scores = site_scores.masked_fill(~connected_to_f_mask, float("-inf"))
            counts = torch.bincount(batch_graph.batch[connected_to_f_mask], minlength=num_graphs)
            has_site = counts > 0

            for gid in range(num_graphs):
                if has_site[gid]:
                    idx = (batch_graph.batch == gid).nonzero(as_tuple=False).view(-1)
                    local_scores = site_scores[idx]
                    best_local = torch.argmax(local_scores)
                    g_site[gid] = x[idx[best_local]]
                else:
                    g_site[gid] = self.null_site_emb

        # Concatenate: [global || selected_site]
        return torch.cat([g_global, g_site], dim=-1)

    def forward(self, batch_graph, col_id, descriptors=None):
        g = self.encode_graph(batch_graph)
        c = self.col_emb(col_id)
        parts = [g, c]
        if self.descriptor_mlp is not None:
            if descriptors is None:
                raise ValueError("descriptors must be provided when descriptor_dim > 0")
            parts.append(self.descriptor_mlp(descriptors))
        h = self.fusion(torch.cat(parts, dim=-1))
        return self.head(h)
