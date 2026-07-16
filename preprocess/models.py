"""
preprocess/models.py  —  Neural network modules for PCA+GNN refinement.

Architecture:
  Data:  Z_data  = P_data  + alpha * GNN_delta(P_data,  graph)
  Query: Z_query = P_query + beta  * MLP_delta(P_query)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedSAGELayer(nn.Module):
    """GraphSAGE aggregation with fixed edge weights from raw-space distances."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.self_lin  = nn.Linear(dim, dim)
        self.neigh_lin = nn.Linear(dim, dim)
        self.bn        = nn.BatchNorm1d(dim)
        self.dropout   = dropout

    def forward(self, x, edge_index, edge_weight):
        src, dst = edge_index[0], edge_index[1]

        msg = x[src] * edge_weight.unsqueeze(1)
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, msg)

        denom = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        denom.index_add_(0, dst, edge_weight.to(x.dtype))
        agg = agg / (denom.unsqueeze(1) + 1e-12)

        h = self.self_lin(x) + self.neigh_lin(agg)
        h = self.bn(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class PCAWeightedResidualGNN(nn.Module):
    """
    PCA-initialized residual GNN for data vectors.

    Z = P + alpha * delta,  where delta is a small GNN correction.
    For angular data (NORMALIZE_OUT=True) output is L2-normalized.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
        res_alpha: float = 0.05,
        normalize_out: bool = False,
    ):
        super().__init__()
        self.res_alpha     = res_alpha
        self.normalize_out = normalize_out

        self.input_proj = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.layers = nn.ModuleList([
            WeightedSAGELayer(hidden_dim, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, p, edge_index, edge_weight):
        h = self.input_proj(p)
        for layer in self.layers:
            h = h + layer(h, edge_index, edge_weight)
        delta = self.delta_head(h)
        z = p + self.res_alpha * delta
        if self.normalize_out:
            z = F.normalize(z, p=2, dim=1)
        return z, delta


class PCAResidualQueryMLP(nn.Module):
    """
    Lightweight MLP residual corrector for query vectors.

    Z_q = P_q + beta * MLP(P_q).
    Trained to retrieve raw-GT data neighbors in the refined Z_data space.
    """

    def __init__(
        self,
        dim: int,
        h1: int = 512,
        h2: int = 256,
        dropout: float = 0.1,
        res_alpha: float = 0.01,
        normalize_out: bool = False,
    ):
        super().__init__()
        self.res_alpha     = res_alpha
        self.normalize_out = normalize_out
        self.net = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.Linear(dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, dim),
        )

    def forward(self, p_q):
        delta = self.net(p_q)
        z_q = p_q + self.res_alpha * delta
        if self.normalize_out:
            z_q = F.normalize(z_q, p=2, dim=1)
        return z_q, delta
