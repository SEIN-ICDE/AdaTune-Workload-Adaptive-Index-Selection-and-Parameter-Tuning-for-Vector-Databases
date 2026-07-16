"""
preprocess/losses.py  —  Loss functions for PCA+GNN training.

All losses work in Euclidean L2 distance space (pairwise_sq_l2).
Both Euclidean and Angular datasets use the same loss functions;
the geometry difference is handled by NORMALIZE_OUT in the model.
"""

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def pairwise_sq_l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a: [B, D], b: [B, M, D] → [B, M]"""
    return ((a.unsqueeze(1) - b) ** 2).sum(dim=-1)


def delta_regularization(delta: torch.Tensor) -> torch.Tensor:
    return delta.pow(2).mean()


# ---------------------------------------------------------------------------
# Data refiner losses
# ---------------------------------------------------------------------------

def weighted_multi_positive_infonce_loss(
    z_anchor: torch.Tensor,
    z_pos: torch.Tensor,
    z_neg: torch.Tensor,
    pos_weights: torch.Tensor,
    temperature: float = 0.15,
) -> torch.Tensor:
    d_pos = pairwise_sq_l2(z_anchor, z_pos)
    d_neg = pairwise_sq_l2(z_anchor, z_neg)
    logits   = torch.cat([-d_pos / temperature, -d_neg / temperature], dim=1)
    log_prob = F.log_softmax(logits, dim=1)
    p = pos_weights.shape[1]
    target_pos = pos_weights / (pos_weights.sum(dim=1, keepdim=True) + 1e-12)
    return -(target_pos * log_prob[:, :p]).sum(dim=1).mean()


def weighted_rank_distillation_loss(
    z_anchor: torch.Tensor,
    z_pos: torch.Tensor,
    raw_pos_dists_sq: torch.Tensor,
    pos_weights: torch.Tensor,
    rank_tau_raw: float = 0.20,
    rank_tau_lat: float = 0.15,
) -> torch.Tensor:
    d_lat = pairwise_sq_l2(z_anchor, z_pos)
    raw_centered = raw_pos_dists_sq - raw_pos_dists_sq[:, :1]
    raw_score    = -raw_centered / rank_tau_raw + torch.log(pos_weights + 1e-12)
    target       = F.softmax(raw_score, dim=1).detach()
    log_prob_lat = F.log_softmax(-d_lat / rank_tau_lat, dim=1)
    return F.kl_div(log_prob_lat, target, reduction="batchmean")


def weighted_local_distance_regression_loss(
    z_anchor: torch.Tensor,
    z_pos: torch.Tensor,
    raw_pos_dists_sq: torch.Tensor,
    pos_weights: torch.Tensor,
) -> torch.Tensor:
    d_lat = pairwise_sq_l2(z_anchor, z_pos)
    raw   = torch.sqrt(torch.clamp(raw_pos_dists_sq, min=1e-12))
    lat   = torch.sqrt(torch.clamp(d_lat,            min=1e-12))
    raw_norm = (raw - raw.mean(dim=1, keepdim=True)) / (raw.std(dim=1, keepdim=True) + 1e-6)
    lat_norm = (lat - lat.mean(dim=1, keepdim=True)) / (lat.std(dim=1, keepdim=True) + 1e-6)
    mse = (lat_norm - raw_norm.detach()).pow(2)
    return (mse * pos_weights).sum(dim=1).div(pos_weights.sum(dim=1) + 1e-12).mean()


def soft_positive_attraction_loss(
    z_anchor: torch.Tensor,
    z_pos: torch.Tensor,
    pos_weights: torch.Tensor,
) -> torch.Tensor:
    d_pos = pairwise_sq_l2(z_anchor, z_pos)
    return (d_pos * pos_weights).sum(dim=1).div(pos_weights.sum(dim=1) + 1e-12).mean()


def compute_data_loss(
    z, delta, p_data_t, anchors_t, pos_t, neg_t,
    raw_pos_dists_sq, pos_weights,
    loss_weights: dict,
    temperature: float = 0.15,
    rank_tau_raw: float = 0.20,
    rank_tau_lat: float = 0.15,
):
    """Aggregate all data-refiner losses into a single scalar."""
    z_anchor = z[anchors_t]
    z_pos    = z[pos_t]
    z_neg    = z[neg_t]

    loss_nce      = weighted_multi_positive_infonce_loss(z_anchor, z_pos, z_neg, pos_weights, temperature)
    loss_rank     = weighted_rank_distillation_loss(z_anchor, z_pos, raw_pos_dists_sq, pos_weights, rank_tau_raw, rank_tau_lat)
    loss_soft_pos = soft_positive_attraction_loss(z_anchor, z_pos, pos_weights)
    loss_dist     = weighted_local_distance_regression_loss(z_anchor, z_pos, raw_pos_dists_sq, pos_weights)
    loss_pca      = F.mse_loss(z_anchor, p_data_t[anchors_t])
    loss_delta    = delta_regularization(delta[anchors_t])

    total = (
        loss_weights["nce"]       * loss_nce      +
        loss_weights["rank"]      * loss_rank     +
        loss_weights["soft_pos"]  * loss_soft_pos +
        loss_weights["dist"]      * loss_dist     +
        loss_weights["pca_align"] * loss_pca      +
        loss_weights["delta"]     * loss_delta
    )
    return total, {
        "total":    total.item(),
        "nce":      loss_nce.item(),
        "rank":     loss_rank.item(),
        "soft_pos": loss_soft_pos.item(),
        "dist":     loss_dist.item(),
        "pca":      loss_pca.item(),
        "delta":    loss_delta.item(),
    }


# ---------------------------------------------------------------------------
# Query MLP losses
# ---------------------------------------------------------------------------

def query_centroid_alignment_loss(
    z_q: torch.Tensor,
    z_pos: torch.Tensor,
    pos_weights: torch.Tensor,
) -> torch.Tensor:
    w        = pos_weights / (pos_weights.sum(dim=1, keepdim=True) + 1e-12)
    centroid = (z_pos * w.unsqueeze(-1)).sum(dim=1)
    return F.mse_loss(z_q, centroid.detach())


def compute_query_loss(
    z_q, delta_q, p_q_batch, pos_t, neg_t, z_data_t,
    query_pos_dists_sq, pos_weights,
    loss_weights: dict,
    temperature: float = 0.15,
    rank_tau_raw: float = 0.20,
    rank_tau_lat: float = 0.15,
):
    """Aggregate all query-MLP losses."""
    z_pos = z_data_t[pos_t]
    z_neg = z_data_t[neg_t]

    loss_nce      = weighted_multi_positive_infonce_loss(z_q, z_pos, z_neg, pos_weights, temperature)
    loss_rank     = weighted_rank_distillation_loss(z_q, z_pos, query_pos_dists_sq, pos_weights, rank_tau_raw, rank_tau_lat)
    loss_centroid = query_centroid_alignment_loss(z_q, z_pos, pos_weights)
    loss_pca      = F.mse_loss(z_q, p_q_batch)
    loss_delta    = delta_regularization(delta_q)

    total = (
        loss_weights["nce"]       * loss_nce      +
        loss_weights["rank"]      * loss_rank     +
        loss_weights["centroid"]  * loss_centroid +
        loss_weights["pca_align"] * loss_pca      +
        loss_weights["delta"]     * loss_delta
    )
    return total, {
        "total":    total.item(),
        "q_nce":    loss_nce.item(),
        "q_rank":   loss_rank.item(),
        "q_cent":   loss_centroid.item(),
        "q_pca":    loss_pca.item(),
        "q_delta":  loss_delta.item(),
    }
