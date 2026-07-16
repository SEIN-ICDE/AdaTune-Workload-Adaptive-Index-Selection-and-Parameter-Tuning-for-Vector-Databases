"""
preprocess/build_dataset.py  —  PCA + Weighted GNN preprocessing.

Produces refined {data, query} vectors in 128-dim and exports them as HDF5
so AdaTune's stage1/stage2 can use them directly as drop-in datasets.

Usage:
  python -m preprocess.build_dataset --dataset gist
  python -m preprocess.build_dataset --dataset wikipedia
  python -m preprocess.build_dataset --dataset gist --no-query-mlp --seed 0

Output (in <result_dir>/<dataset>/):
  <prefix>_refined_data.npy       : z_data  [N, 128] float32
  <prefix>_pca_query.npy          : p_query [Q, 128] float32  (PCA-only)
  <prefix>_mlp_query.npy          : z_query [Q, 128] float32  (MLP-corrected, if trained)
  <prefix>_summary.csv            : recall / LID summary
  <dataset>-pcagnn-128-{metric}.hdf5 : HDF5 for AdaTune (train=z_data, test=z_query_best)
"""

import argparse
import copy
import os
import pickle
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import faiss
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA

from .models import PCAResidualQueryMLP, PCAWeightedResidualGNN
from .losses import compute_data_loss, compute_query_loss

# ============================================================
# Dataset presets
# ============================================================

@dataclass
class DatasetConfig:
    name: str
    raw_path: str            # path to the raw HDF5 (or npy dir for wiki)
    raw_format: str          # "hdf5" | "wiki_npy"
    metric: str              # "L2" | "IP"
    weight_mode: str         # "dist" | "cosine"
    normalize_out: bool      # True for angular data
    n_sample: int            # number of base vectors to use
    k_graph: int             # kNN graph degree
    k_eval: int              # eval neighbor count
    hidden_dim: int          # GNN hidden size
    res_alpha: float         # GNN residual scale
    neg_start: int
    neg_end: int
    n_hard_neg: int
    query_train_k: int       # how many raw query neighbors to retrieve
    query_neg_start: int
    query_neg_end: int
    result_dir: str
    pca_dim: int = 128
    k_pos: int = 5
    seed: int = 42
    n_gnn_layers: int = 2
    dropout: float = 0.10
    query_res_alpha: float = 0.01
    gnn_epochs: int = 200
    steps_per_epoch: int = 80
    anchor_batch: int = 1024
    gnn_lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 5.0
    eval_every: int = 20
    eval_subset: int = 5000
    qe_epochs: int = 150
    qe_batch: int = 256
    qe_lr: float = 1e-3
    qe_hidden1: int = 512
    qe_hidden2: int = 256
    query_train_ratio: float = 0.60
    query_val_ratio: float = 0.20
    min_edge_weight: float = 0.05
    edge_temp: float = 0.08
    temperature: float = 0.15
    rank_tau_raw: float = 0.20
    rank_tau_lat: float = 0.15
    data_loss_weights: Dict[str, float] = field(default_factory=lambda: {
        "nce": 0.20, "rank": 0.10, "soft_pos": 0.10,
        "dist": 0.05, "pca_align": 5.00, "delta": 0.10,
    })
    query_loss_weights: Dict[str, float] = field(default_factory=lambda: {
        "nce": 1.00, "rank": 0.20, "centroid": 0.05,
        "pca_align": 5.00, "delta": 0.10,
    })
    eval_k_list: List[int] = field(default_factory=lambda: [5, 10, 15])
    lid_k: int = 20


PRESETS: Dict[str, DatasetConfig] = {
    "gist": DatasetConfig(
        name="gist",
        raw_path="/home/sein/Vector_DBMS/dataset/gist-960-euclidean.hdf5",
        raw_format="hdf5",
        metric="L2",
        weight_mode="dist",
        normalize_out=False,
        n_sample=1_000_000,
        k_graph=10,
        k_eval=100,
        hidden_dim=32,
        res_alpha=0.02,
        neg_start=32,
        neg_end=100,
        n_hard_neg=16,
        query_train_k=300,
        query_neg_start=64,
        query_neg_end=200,
        result_dir="/home/sein/Vector_DBMS/AdaTune/preprocess_output/gist",
    ),
    "wikipedia": DatasetConfig(
        name="wikipedia",
        raw_path="/home/sein/Vector_DBMS/Hard_data/wiki_embeddings",
        raw_format="wiki_npy",
        metric="IP",
        weight_mode="cosine",
        normalize_out=True,
        n_sample=100_000,
        k_graph=20,
        k_eval=300,
        hidden_dim=256,
        res_alpha=0.05,
        neg_start=64,
        neg_end=200,
        n_hard_neg=32,
        query_train_k=100,
        query_neg_start=20,
        query_neg_end=100,
        result_dir="/home/sein/Vector_DBMS/AdaTune/preprocess_output/wikipedia",
    ),
}


# ============================================================
# Helpers
# ============================================================

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def _cache(cache_dir: str, tag: str, cfg: DatasetConfig) -> str:
    return os.path.join(cache_dir, f"{cfg.name}_seed{cfg.seed}_{tag}.npy")


# ============================================================
# 1. Load raw data
# ============================================================

def load_raw_data(cfg: DatasetConfig, cache_dir: str):
    c_xb  = _cache(cache_dir, "xb", cfg)
    c_xq  = _cache(cache_dir, "xq", cfg)
    c_qgt = _cache(cache_dir, f"query_gt_k{cfg.k_pos * 4}", cfg)  # just reuse k_pos*4 as GT
    c_qnb = _cache(cache_dir, f"query_nb_k{cfg.query_train_k}", cfg)
    c_qds = _cache(cache_dir, f"query_ds_k{cfg.query_train_k}", cfg)
    caches = [c_xb, c_xq, c_qgt, c_qnb, c_qds]

    if all(os.path.exists(p) for p in caches):
        print("[1] Loading from cache...")
        xb = np.load(c_xb)
        xq = np.load(c_xq)
        query_gt              = np.load(c_qgt)
        query_neighbors_train = np.load(c_qnb)
        query_dists_sq_train  = np.load(c_qds)
        print(f"    xb={xb.shape}  xq={xq.shape}")
        return xb, xq, query_gt, query_neighbors_train, query_dists_sq_train

    print("[1] Loading raw data...")
    if cfg.raw_format == "hdf5":
        with h5py.File(cfg.raw_path, "r") as f:
            total = f["train"].shape[0]
            rng = np.random.default_rng(cfg.seed)
            idx = np.sort(rng.choice(total, size=cfg.n_sample, replace=False))
            xb  = f["train"][idx].astype("float32")
            xq  = f["test"][:].astype("float32")

        # Remove near-zero-norm vectors (degenerate hub nodes)
        norms = np.linalg.norm(xb, axis=1)
        valid = norms > norms.mean() * 0.1
        if not valid.all():
            print(f"    Removing {(~valid).sum()} near-zero-norm vectors")
            xb = xb[valid]

    elif cfg.raw_format == "wiki_npy":
        wiki_dir = cfg.raw_path
        xb  = np.load(os.path.join(wiki_dir, "wiki_xb_100k_768.npy")).astype("float32")
        xq  = np.load(os.path.join(wiki_dir, "wiki_xq_10k_768.npy")).astype("float32")
        gt  = np.load(os.path.join(wiki_dir, "wiki_gt_k100.npy")).astype("int64")
        # Already L2-normalized; re-normalize for safety
        xb  = xb / (np.linalg.norm(xb, axis=1, keepdims=True) + 1e-12)
        xq  = xq / (np.linalg.norm(xq, axis=1, keepdims=True) + 1e-12)
        # Precomputed GT (IndexFlatIP on unit vectors)
        query_neighbors_train = gt[:, :cfg.query_train_k]
        query_gt_out = gt[:, :20]
        # Derive L2-sq distances from cosine similarities
        print("    Computing query L2-sq distances from GT...")
        query_dists_sq_train = np.empty((len(xq), cfg.query_train_k), dtype=np.float32)
        chunk = 512
        for s in range(0, len(xq), chunk):
            e = min(s + chunk, len(xq))
            sims = np.einsum("cd,cmd->cm", xq[s:e], xb[query_neighbors_train[s:e]])
            query_dists_sq_train[s:e] = 2.0 - 2.0 * sims
        for p in [c_xb, c_xq, c_qgt, c_qnb, c_qds]:
            pass  # will save below
        np.save(c_xb,  xb)
        np.save(c_xq,  xq)
        np.save(c_qgt, query_gt_out.astype(np.int64))
        np.save(c_qnb, query_neighbors_train.astype(np.int64))
        np.save(c_qds, query_dists_sq_train.astype(np.float32))
        print(f"    xb={xb.shape}  xq={xq.shape}")
        return xb, xq, query_gt_out.astype(np.int64), query_neighbors_train.astype(np.int64), query_dists_sq_train
    else:
        raise ValueError(f"Unknown raw_format: {cfg.raw_format}")

    print(f"    xb={xb.shape}  xq={xq.shape}")
    print(f"    Computing raw query neighbors (k={cfg.query_train_k})...")
    faiss_metric = faiss.METRIC_L2 if cfg.metric == "L2" else faiss.METRIC_INNER_PRODUCT
    index_gt = faiss.IndexFlat(xb.shape[1], faiss_metric)
    index_gt.add(np.ascontiguousarray(xb))
    D_sq, I = index_gt.search(np.ascontiguousarray(xq), cfg.query_train_k)
    if cfg.metric == "IP":
        D_sq = 2.0 - 2.0 * D_sq.astype(np.float32)  # cosine sim → L2-sq approx
    query_neighbors_train = I.astype(np.int64)
    query_dists_sq_train  = D_sq.astype(np.float32)
    query_gt_k = 20
    query_gt   = query_neighbors_train[:, :query_gt_k]

    np.save(c_xb,  xb)
    np.save(c_xq,  xq)
    np.save(c_qgt, query_gt.astype(np.int64))
    np.save(c_qnb, query_neighbors_train.astype(np.int64))
    np.save(c_qds, query_dists_sq_train.astype(np.float32))
    print("    Cached.")
    return xb, xq, query_gt.astype(np.int64), query_neighbors_train.astype(np.int64), query_dists_sq_train


# ============================================================
# 2. kNN graph + fixed edge weights
# ============================================================

def _cosine_sim(xb: np.ndarray, neighbors: np.ndarray, chunk: int = 256) -> np.ndarray:
    xb_n = xb / (np.linalg.norm(xb, axis=1, keepdims=True) + 1e-12)
    N, K = neighbors.shape
    out  = np.empty((N, K), dtype=np.float32)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        out[s:e] = np.einsum("bd,bkd->bk", xb_n[s:e], xb_n[neighbors[s:e]])
    return out


def _weights_from_dists(full_dists_sq: np.ndarray, k_graph: int, min_w: float) -> np.ndarray:
    d = np.sqrt(np.maximum(full_dists_sq, 0.0)).astype(np.float32)
    d0 = d[:, :1]
    centered = d - d0
    scale = np.mean(np.maximum(centered[:, 1:k_graph], 1e-6), axis=1, keepdims=True) if k_graph > 1 \
            else np.ones((d.shape[0], 1), dtype=np.float32)
    scale = np.maximum(scale, 1e-6)
    return np.maximum(np.exp(-centered / scale).astype(np.float32), min_w)


def _weights_from_cosine(xb: np.ndarray, neighbors: np.ndarray, k_graph: int,
                          min_w: float, edge_temp: float) -> np.ndarray:
    sims      = _cosine_sim(xb, neighbors)
    threshold = sims[:, :k_graph].mean(axis=1, keepdims=True)
    return np.maximum(sigmoid_np((sims - threshold) / edge_temp).astype(np.float32), min_w)


def build_raw_graph(xb: np.ndarray, cfg: DatasetConfig, cache_dir: str):
    k_search = max(cfg.k_graph, cfg.k_eval, cfg.neg_end) + 1
    c_eval = _cache(cache_dir, f"eval_nb_k{cfg.k_eval}_kg{cfg.k_graph}", cfg)
    c_full = _cache(cache_dir, f"full_nb_kg{cfg.k_graph}_ne{cfg.neg_end}", cfg)
    c_dsq  = _cache(cache_dir, f"full_dists_kg{cfg.k_graph}_ne{cfg.neg_end}", cfg)
    c_fw   = _cache(cache_dir, f"full_wts_{cfg.weight_mode}_kg{cfg.k_graph}", cfg)
    c_ei   = _cache(cache_dir, f"edge_index_kg{cfg.k_graph}", cfg)
    c_ew   = _cache(cache_dir, f"edge_weight_{cfg.weight_mode}_kg{cfg.k_graph}", cfg)

    if all(os.path.exists(p) for p in [c_eval, c_full, c_dsq, c_fw, c_ei, c_ew]):
        print(f"\n[2] Loading kNN graph from cache...")
        eval_neighbors = np.load(c_eval)
        full_neighbors = np.load(c_full)
        full_dists_sq  = np.load(c_dsq)
        full_weights   = np.load(c_fw)
        edge_index  = torch.tensor(np.load(c_ei), dtype=torch.long)
        edge_weight = torch.tensor(np.load(c_ew), dtype=torch.float32)
        print(f"    nodes={xb.shape[0]:,}, edges={edge_index.shape[1]:,}")
        return eval_neighbors, full_neighbors, full_dists_sq, full_weights, edge_index, edge_weight

    print(f"\n[2] Building kNN graph (K_GRAPH={cfg.k_graph}, K_EVAL={cfg.k_eval})...")
    n, d = xb.shape
    t0 = time.time()

    # Prefer GPU FAISS for large datasets
    try:
        res = faiss.StandardGpuResources()
        gpu_idx = faiss.GpuIndexFlatL2(res, d, faiss.GpuIndexFlatConfig())
        gpu_idx.add(np.ascontiguousarray(xb))
        D_sq, I = gpu_idx.search(np.ascontiguousarray(xb), k_search)
        print(f"    kNN done (GPU): {time.time() - t0:.1f}s")
        del gpu_idx, res
        torch.cuda.empty_cache()
    except Exception as exc:
        print(f"    GPU kNN failed ({exc}), falling back to CPU IVF...")
        nlist = min(4096, int(np.sqrt(n)))
        quantizer = faiss.IndexFlatL2(d)
        ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        ivf.train(np.ascontiguousarray(xb))
        ivf.add(np.ascontiguousarray(xb))
        ivf.nprobe = 64
        D_sq, I = ivf.search(np.ascontiguousarray(xb), k_search)
        print(f"    kNN done (CPU IVF): {time.time() - t0:.1f}s")

    D_sq = D_sq[:, 1:].astype(np.float32)
    I    = I[:, 1:].astype(np.int64)

    eval_neighbors = I[:, :cfg.k_eval]
    full_neighbors = I
    full_dists_sq  = D_sq

    if cfg.weight_mode == "cosine":
        full_weights = _weights_from_cosine(xb, full_neighbors, cfg.k_graph,
                                             cfg.min_edge_weight, cfg.edge_temp)
    else:
        full_weights = _weights_from_dists(full_dists_sq, cfg.k_graph, cfg.min_edge_weight)

    src = np.repeat(np.arange(n), cfg.k_graph)
    dst = I[:, :cfg.k_graph].reshape(-1)
    ew  = full_weights[:, :cfg.k_graph].reshape(-1)

    ei_np = np.stack([src, dst], axis=0)
    ei_np = np.concatenate([ei_np, ei_np[[1, 0]]], axis=1)
    ew_np = np.concatenate([ew, ew]).astype(np.float32)

    edge_index  = torch.tensor(ei_np, dtype=torch.long)
    edge_weight = torch.tensor(ew_np, dtype=torch.float32)
    print(f"    nodes={n:,}, edges={edge_index.shape[1]:,}  "
          f"w mean={ew_np.mean():.4f} min={ew_np.min():.4f}")

    np.save(c_eval, eval_neighbors)
    np.save(c_full, full_neighbors)
    np.save(c_dsq,  full_dists_sq)
    np.save(c_fw,   full_weights)
    np.save(c_ei,   ei_np)
    np.save(c_ew,   ew_np)
    print("    Cached.")
    return eval_neighbors, full_neighbors, full_dists_sq, full_weights, edge_index, edge_weight


# ============================================================
# 3. PCA
# ============================================================

def fit_pca(xb: np.ndarray, xq: np.ndarray, cfg: DatasetConfig, cache_dir: str):
    c_data  = _cache(cache_dir, f"pca{cfg.pca_dim}_data", cfg)
    c_query = _cache(cache_dir, f"pca{cfg.pca_dim}_query", cfg)
    c_model = os.path.join(cache_dir, f"{cfg.name}_seed{cfg.seed}_pca{cfg.pca_dim}.pkl")

    if all(os.path.exists(p) for p in [c_data, c_query, c_model]):
        print(f"\n[3] Loading PCA-{cfg.pca_dim} from cache...")
        with open(c_model, "rb") as f:
            pca = pickle.load(f)
        return pca, np.load(c_data), np.load(c_query)

    print(f"\n[3] Fitting PCA ({xb.shape[1]} → {cfg.pca_dim})...")
    t0  = time.time()
    pca = PCA(n_components=cfg.pca_dim, svd_solver="randomized", random_state=cfg.seed)
    p_data  = pca.fit_transform(xb).astype("float32")
    p_query = pca.transform(xq).astype("float32")
    if cfg.normalize_out:  # angular: project PCA onto unit sphere for IP search
        p_data  = p_data  / (np.linalg.norm(p_data,  axis=1, keepdims=True) + 1e-12)
        p_query = p_query / (np.linalg.norm(p_query, axis=1, keepdims=True) + 1e-12)
    print(f"    done: {time.time() - t0:.1f}s  "
          f"var_explained={pca.explained_variance_ratio_.sum():.4f}")

    np.save(c_data,  p_data)
    np.save(c_query, p_query)
    with open(c_model, "wb") as f:
        pickle.dump(pca, f)
    return pca, p_data, p_query


# ============================================================
# 4. Training helpers
# ============================================================

def _pos_neg_indices(anchors, full_neighbors, cfg, rng):
    pos_idx = full_neighbors[anchors, :cfg.k_pos].astype(np.int64)
    neg_cands = full_neighbors[anchors, cfg.neg_start:cfg.neg_end]
    B = len(anchors)
    neg_idx = np.empty((B, cfg.n_hard_neg), dtype=np.int64)
    for i in range(B):
        cand    = neg_cands[i]
        replace = len(cand) < cfg.n_hard_neg
        neg_idx[i] = rng.choice(cand, size=cfg.n_hard_neg, replace=replace)
    return pos_idx, neg_idx


def _edge_pos_weights(dists_sq, k_pos, min_w):
    raw = dists_sq[:, :k_pos].astype(np.float32)
    centered = raw - raw[:, :1]
    scale = np.mean(np.maximum(centered[:, 1:], 1e-6), axis=1, keepdims=True) if k_pos > 1 \
            else np.ones((raw.shape[0], 1), dtype=np.float32)
    return np.maximum(np.exp(-centered / np.maximum(scale, 1e-6)).astype(np.float32), min_w)


def _query_pos_neg(query_nb_batch, cfg, rng):
    B = query_nb_batch.shape[0]
    pos_idx = query_nb_batch[:, :cfg.k_pos].astype(np.int64)
    neg_idx = np.empty((B, cfg.n_hard_neg), dtype=np.int64)
    for i in range(B):
        cand    = query_nb_batch[i, cfg.query_neg_start:cfg.query_neg_end]
        if len(cand) == 0:
            cand = query_nb_batch[i, cfg.k_pos:]
        replace = len(cand) < cfg.n_hard_neg
        neg_idx[i] = rng.choice(cand, size=cfg.n_hard_neg, replace=replace)
    return pos_idx, neg_idx


def _faiss_index(z: np.ndarray, metric: str):
    if metric == "IP":
        idx = faiss.IndexFlatIP(z.shape[1])
    else:
        idx = faiss.IndexFlatL2(z.shape[1])
    idx.add(np.ascontiguousarray(z.astype("float32")))
    return idx


def _recall_at_k(gt, pred, k):
    return float(np.mean([
        len(np.intersect1d(gt[i, :k], pred[i, :k])) / k
        for i in range(len(gt))
    ]))


def _quick_subset_recall(z_np, eval_neighbors, cfg, k=10):
    rng = np.random.default_rng(cfg.seed)
    N   = z_np.shape[0]
    sub = rng.choice(N, size=min(cfg.eval_subset, N), replace=False)
    idx = _faiss_index(z_np, cfg.metric)
    _, I = idx.search(z_np[sub].astype("float32"), k + 1)
    vals = [
        len(np.intersect1d(eval_neighbors[i, :k], I[row, 1:])) / k
        for row, i in enumerate(sub)
    ]
    return float(np.mean(vals))


# ============================================================
# 5. GNN training loop
# ============================================================

def train_data_refiner(model, p_data, edge_index, edge_weight,
                       full_neighbors, full_dists_sq, full_weights,
                       eval_neighbors, cfg, device):
    p_data_t    = torch.tensor(p_data,  dtype=torch.float32, device=device)
    edge_index  = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.gnn_lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.gnn_epochs)
    rng = np.random.default_rng(cfg.seed)
    N   = p_data.shape[0]

    best_score, best_state = -1.0, None

    print(f"\n[4] Training GNN refiner ({cfg.gnn_epochs} epochs, "
          f"res_alpha={cfg.res_alpha}, normalize_out={cfg.normalize_out})")

    for epoch in range(1, cfg.gnn_epochs + 1):
        model.train()
        running = {k: 0.0 for k in ["total", "nce", "rank", "soft_pos", "dist", "pca", "delta"]}

        for _ in range(cfg.steps_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            z, delta = model(p_data_t, edge_index, edge_weight)

            anchors = rng.choice(N, size=min(cfg.anchor_batch, N), replace=False)
            pos_np, neg_np = _pos_neg_indices(anchors, full_neighbors, cfg, rng)

            anchors_t = torch.tensor(anchors,   dtype=torch.long, device=device)
            pos_t     = torch.tensor(pos_np,    dtype=torch.long, device=device)
            neg_t     = torch.tensor(neg_np,    dtype=torch.long, device=device)
            raw_dists = torch.tensor(full_dists_sq[anchors, :cfg.k_pos], dtype=torch.float32, device=device)
            pos_w     = torch.tensor(full_weights[anchors,  :cfg.k_pos], dtype=torch.float32, device=device)

            loss, ld = compute_data_loss(
                z=z, delta=delta, p_data_t=p_data_t,
                anchors_t=anchors_t, pos_t=pos_t, neg_t=neg_t,
                raw_pos_dists_sq=raw_dists, pos_weights=pos_w,
                loss_weights=cfg.data_loss_weights,
                temperature=cfg.temperature,
                rank_tau_raw=cfg.rank_tau_raw,
                rank_tau_lat=cfg.rank_tau_lat,
            )
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            for k in running:
                running[k] += ld[k]

        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            msg = "  ".join(f"{k}={running[k] / cfg.steps_per_epoch:.4f}" for k in running)
            print(f"    epoch {epoch:3d}/{cfg.gnn_epochs}  {msg}  lr={scheduler.get_last_lr()[0]:.6f}")

        if epoch % cfg.eval_every == 0 or epoch == cfg.gnn_epochs:
            model.eval()
            with torch.no_grad():
                z_np = model(p_data_t, edge_index, edge_weight)[0].cpu().numpy().astype("float32")
            r10 = _quick_subset_recall(z_np, eval_neighbors, cfg)
            print(f"        [eval] subset Recall@10 = {r10:.4f}")
            if r10 > best_score:
                best_score = r10
                best_state = copy.deepcopy(model.state_dict())
                print(f"        → new best: {best_score:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"    Best model restored: Recall@10={best_score:.4f}")
    return model


# ============================================================
# 6. Query MLP training loop
# ============================================================

def train_query_mlp(qe, p_query, z_data, query_neighbors_train, query_dists_sq_train,
                    query_gt, train_idx, val_idx, cfg, device):
    p_query_t = torch.tensor(p_query, dtype=torch.float32, device=device)
    z_data_t  = torch.tensor(z_data,  dtype=torch.float32, device=device)

    optimizer = torch.optim.AdamW(qe.parameters(), lr=cfg.qe_lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.qe_epochs)
    rng = np.random.default_rng(cfg.seed)

    train_idx = np.asarray(train_idx, dtype=np.int64)
    val_idx   = np.asarray(val_idx,   dtype=np.int64)

    best_r10, best_state = -1.0, None

    print(f"\n[5] Training query MLP ({cfg.qe_epochs} epochs, "
          f"query_res_alpha={cfg.query_res_alpha})")
    print(f"    split: train={len(train_idx)}, val={len(val_idx)}")

    for epoch in range(1, cfg.qe_epochs + 1):
        qe.train()
        perm    = rng.permutation(train_idx)
        running = {k: 0.0 for k in ["total", "q_nce", "q_rank", "q_cent", "q_pca", "q_delta"]}
        n_batch = 0

        for s in range(0, len(perm), cfg.qe_batch):
            bidx = perm[s:s + cfg.qe_batch]
            optimizer.zero_grad(set_to_none=True)

            p_q_batch = p_query_t[bidx]
            z_q, dq   = qe(p_q_batch)

            pos_np, neg_np = _query_pos_neg(query_neighbors_train[bidx], cfg, rng)
            pos_w_np       = _edge_pos_weights(query_dists_sq_train[bidx], cfg.k_pos, cfg.min_edge_weight)

            pos_t    = torch.tensor(pos_np, dtype=torch.long, device=device)
            neg_t    = torch.tensor(neg_np, dtype=torch.long, device=device)
            pos_w    = torch.tensor(pos_w_np, dtype=torch.float32, device=device)
            qdists   = torch.tensor(query_dists_sq_train[bidx, :cfg.k_pos], dtype=torch.float32, device=device)

            loss, ld = compute_query_loss(
                z_q=z_q, delta_q=dq, p_q_batch=p_q_batch,
                pos_t=pos_t, neg_t=neg_t, z_data_t=z_data_t,
                query_pos_dists_sq=qdists, pos_weights=pos_w,
                loss_weights=cfg.query_loss_weights,
                temperature=cfg.temperature,
                rank_tau_raw=cfg.rank_tau_raw,
                rank_tau_lat=cfg.rank_tau_lat,
            )
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(qe.parameters(), cfg.grad_clip)
            optimizer.step()
            for k in running:
                running[k] += ld[k]
            n_batch += 1

        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            denom = max(n_batch, 1)
            msg = "  ".join(f"{k}={running[k] / denom:.4f}" for k in running)
            print(f"    epoch {epoch:3d}/{cfg.qe_epochs}  {msg}  lr={scheduler.get_last_lr()[0]:.6f}")

        if len(val_idx) > 0 and (epoch % 10 == 0 or epoch == cfg.qe_epochs):
            qe.eval()
            with torch.no_grad():
                z_val = qe(p_query_t[val_idx])[0].cpu().numpy().astype("float32")
            pred_val = _faiss_index(z_data, cfg.metric)
            _, I_val = pred_val.search(z_val.astype("float32"), max(cfg.eval_k_list))
            val_r10  = _recall_at_k(query_gt[val_idx], I_val, k=10)
            print(f"        [val] Query Recall@10 = {val_r10:.4f}")
            if val_r10 > best_r10:
                best_r10  = val_r10
                best_state = copy.deepcopy(qe.state_dict())
                print(f"        → new best: {best_r10:.4f}")

    if best_state is not None:
        qe.load_state_dict(best_state)
        print(f"    Best query MLP restored: val Recall@10={best_r10:.4f}")
    return qe


# ============================================================
# 7. LID
# ============================================================

def compute_lid(X: np.ndarray, k: int = 20, eps: float = 1e-12) -> np.ndarray:
    X = np.ascontiguousarray(X.astype("float32"))
    idx = faiss.IndexFlatL2(X.shape[1])
    idx.add(X)
    D_sq, _ = idx.search(X, k + 1)
    D = np.sqrt(np.maximum(D_sq[:, 1:], 0.0))
    D = np.maximum(D, eps)
    r_k = np.maximum(D[:, -1:], eps)
    ratios = np.clip(D / r_k, eps, 1.0)
    denom  = np.log(ratios).sum(axis=1)
    lid    = np.full(X.shape[0], np.nan, dtype=np.float32)
    valid  = denom < -1e-6
    lid[valid] = (-k / denom[valid]).astype(np.float32)
    return lid


# ============================================================
# 8. HDF5 export
# ============================================================

def export_hdf5(z_data, z_query, query_gt, cfg, out_dir):
    metric_tag = "euclidean" if cfg.metric == "L2" else "angular"
    fname = os.path.join(out_dir, f"{cfg.name}-pcagnn-{cfg.pca_dim}-{metric_tag}.hdf5")
    with h5py.File(fname, "w") as f:
        f.create_dataset("train",     data=z_data.astype("float32"),  compression="gzip")
        f.create_dataset("test",      data=z_query.astype("float32"), compression="gzip")
        f.create_dataset("neighbors", data=query_gt.astype("int32"),  compression="gzip")
        f.attrs["metric"]  = cfg.metric
        f.attrs["dataset"] = cfg.name
    print(f"    Exported HDF5 → {fname}")
    return fname


# ============================================================
# MAIN
# ============================================================

def main(cfg: DatasetConfig, use_query_mlp: bool = True):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(cfg.result_dir, exist_ok=True)
    cache_dir = os.path.join(cfg.result_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    t0 = time.time()
    print("=" * 80)
    print(f"  AdaTune PCA+GNN preprocessing  |  dataset={cfg.name}")
    print(f"  n_sample={cfg.n_sample:,}  pca_dim={cfg.pca_dim}  k_graph={cfg.k_graph}")
    print(f"  weight_mode={cfg.weight_mode}  normalize_out={cfg.normalize_out}")
    print(f"  res_alpha={cfg.res_alpha}  hidden_dim={cfg.hidden_dim}")
    print("=" * 80)

    # ---------- 1. Data ----------
    xb, xq, query_gt, query_neighbors_train, query_dists_sq_train = load_raw_data(cfg, cache_dir)

    # ---------- 2. Graph ----------
    eval_neighbors, full_neighbors, full_dists_sq, full_weights, edge_index, edge_weight = \
        build_raw_graph(xb, cfg, cache_dir)

    # ---------- 3. PCA ----------
    _, p_data, p_query = fit_pca(xb, xq, cfg, cache_dir)

    # ---------- Query split (MLP only) ----------
    rng_split = np.random.default_rng(cfg.seed)
    perm = rng_split.permutation(len(xq))
    n_train = int(len(xq) * cfg.query_train_ratio)
    n_val   = int(len(xq) * cfg.query_val_ratio)
    train_idx = np.sort(perm[:n_train]).astype(np.int64)
    val_idx   = np.sort(perm[n_train:n_train + n_val]).astype(np.int64)
    test_idx  = np.sort(perm[n_train + n_val:]).astype(np.int64)

    # ---------- 4. PCA baseline eval ----------
    print("\n[PCA baseline]")
    pca_idx = _faiss_index(p_data, cfg.metric)
    _, pca_self = pca_idx.search(p_data.astype("float32"), max(cfg.eval_k_list) + 1)
    pca_self = pca_self[:, 1:]
    _, pca_q_pred = _faiss_index(p_data, cfg.metric).search(p_query.astype("float32"), max(cfg.eval_k_list))
    lid_pca = compute_lid(p_data, k=cfg.lid_k)
    for k in cfg.eval_k_list:
        r = _recall_at_k(eval_neighbors, pca_self, k)
        print(f"    PCA Data  Recall@{k:2d}: {r:.4f}")
    for k in cfg.eval_k_list:
        r = _recall_at_k(query_gt, pca_q_pred, k)
        print(f"    PCA Query Recall@{k:2d}: {r:.4f}")
    print(f"    PCA LID median: {np.nanmedian(lid_pca):.3f}")

    # ---------- 5. GNN ----------
    model = PCAWeightedResidualGNN(
        dim=cfg.pca_dim, hidden_dim=cfg.hidden_dim, n_layers=cfg.n_gnn_layers,
        dropout=cfg.dropout, res_alpha=cfg.res_alpha, normalize_out=cfg.normalize_out,
    ).to(device)
    print(f"\n  GNN params: {sum(p.numel() for p in model.parameters()):,}")
    model = train_data_refiner(
        model, p_data, edge_index, edge_weight,
        full_neighbors, full_dists_sq, full_weights, eval_neighbors, cfg, device,
    )
    model.eval()
    with torch.no_grad():
        p_t  = torch.tensor(p_data, dtype=torch.float32, device=device)
        ei_t = edge_index.to(device)
        ew_t = edge_weight.to(device)
        z_data_t, delta_t = model(p_t, ei_t, ew_t)
        z_data     = z_data_t.cpu().numpy().astype("float32")
        delta_data = delta_t.cpu().numpy().astype("float32")

    # ---------- 6. Refined data eval ----------
    print("\n[Refined data]")
    ref_idx = _faiss_index(z_data, cfg.metric)
    _, ref_self = ref_idx.search(z_data.astype("float32"), max(cfg.eval_k_list) + 1)
    ref_self = ref_self[:, 1:]
    lid_ref  = compute_lid(z_data, k=cfg.lid_k)
    for k in cfg.eval_k_list:
        r = _recall_at_k(eval_neighbors, ref_self, k)
        print(f"    Refined Data Recall@{k:2d}: {r:.4f}")
    print(f"    Refined LID median: {np.nanmedian(lid_ref):.3f}")

    # ---------- 7. Query mode A: PCA query ----------
    z_query_pca = p_query.astype("float32")
    _, pred_pca = _faiss_index(z_data, cfg.metric).search(z_query_pca, max(cfg.eval_k_list))
    print("\n[Query mode A: PCA query → refined data]")
    for k in cfg.eval_k_list:
        r = _recall_at_k(query_gt, pred_pca, k)
        print(f"    Recall@{k:2d}: {r:.4f}")

    # ---------- 8. Query mode B: MLP ----------
    z_query_mlp = None
    pred_mlp    = None
    if use_query_mlp:
        qe = PCAResidualQueryMLP(
            dim=cfg.pca_dim, h1=cfg.qe_hidden1, h2=cfg.qe_hidden2,
            dropout=cfg.dropout, res_alpha=cfg.query_res_alpha, normalize_out=cfg.normalize_out,
        ).to(device)
        print(f"  Query MLP params: {sum(p.numel() for p in qe.parameters()):,}")
        qe = train_query_mlp(
            qe=qe, p_query=p_query, z_data=z_data,
            query_neighbors_train=query_neighbors_train,
            query_dists_sq_train=query_dists_sq_train,
            query_gt=query_gt, train_idx=train_idx, val_idx=val_idx,
            cfg=cfg, device=device,
        )
        qe.eval()
        with torch.no_grad():
            z_query_mlp = qe(torch.tensor(p_query, dtype=torch.float32, device=device))[0].cpu().numpy().astype("float32")
        _, pred_mlp = _faiss_index(z_data, cfg.metric).search(z_query_mlp, max(cfg.eval_k_list))
        print("\n[Query mode B: MLP query → refined data]")
        for k in cfg.eval_k_list:
            r = _recall_at_k(query_gt[test_idx], pred_mlp[test_idx], k)
            print(f"    Recall@{k:2d} (test): {r:.4f}")

    # ---------- 9. Save outputs ----------
    prefix = f"{cfg.name}_pca{cfg.pca_dim}_resalpha{cfg.res_alpha}_qalpha{cfg.query_res_alpha}"
    out    = cfg.result_dir

    np.save(os.path.join(out, f"{prefix}_pca_data.npy"),     p_data)
    np.save(os.path.join(out, f"{prefix}_pca_query.npy"),    p_query)
    np.save(os.path.join(out, f"{prefix}_refined_data.npy"), z_data)
    np.save(os.path.join(out, f"{prefix}_delta_data.npy"),   delta_data)
    np.save(os.path.join(out, f"{prefix}_query_pca.npy"),    z_query_pca)
    np.save(os.path.join(out, f"{prefix}_query_train_idx.npy"), train_idx)
    np.save(os.path.join(out, f"{prefix}_query_val_idx.npy"),   val_idx)
    np.save(os.path.join(out, f"{prefix}_query_test_idx.npy"),  test_idx)

    if z_query_mlp is not None:
        np.save(os.path.join(out, f"{prefix}_query_mlp.npy"), z_query_mlp)

    # Best query: prefer MLP if available
    z_query_best = z_query_mlp if z_query_mlp is not None else z_query_pca

    # HDF5 for AdaTune
    hdf5_path = export_hdf5(z_data, z_query_best, query_gt, cfg, out)

    # Summary CSV
    rows = []
    for k in cfg.eval_k_list:
        rows.append({
            "method": "PCA",
            "split": "all",
            "k": k,
            "data_recall":  _recall_at_k(eval_neighbors, pca_self, k),
            "query_recall": _recall_at_k(query_gt, pca_q_pred, k),
            "lid_median":   float(np.nanmedian(lid_pca)),
        })
        rows.append({
            "method": "GNN+PCA_query",
            "split": "all",
            "k": k,
            "data_recall":  _recall_at_k(eval_neighbors, ref_self, k),
            "query_recall": _recall_at_k(query_gt, pred_pca, k),
            "lid_median":   float(np.nanmedian(lid_ref)),
        })
        if pred_mlp is not None:
            rows.append({
                "method": "GNN+MLP_query",
                "split": "test",
                "k": k,
                "data_recall":  _recall_at_k(eval_neighbors, ref_self, k),
                "query_recall": _recall_at_k(query_gt[test_idx], pred_mlp[test_idx], k),
                "lid_median":   float(np.nanmedian(lid_ref)),
            })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(out, f"{prefix}_summary.csv")
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print(df.to_string(index=False))
    print(f"\nHDF5  → {hdf5_path}")
    print(f"CSV   → {csv_path}")
    print(f"Total time: {(time.time() - t0) / 60:.1f} min")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AdaTune PCA+GNN preprocessing")
    parser.add_argument("--dataset",      required=True, choices=list(PRESETS.keys()))
    parser.add_argument("--no-query-mlp", action="store_true", help="Skip query MLP training")
    parser.add_argument("--seed",         type=int, default=None)
    parser.add_argument("--result-dir",   default=None, help="Override output directory")
    args = parser.parse_args()

    cfg = PRESETS[args.dataset]
    if args.seed is not None:
        cfg.seed = args.seed
    if args.result_dir is not None:
        cfg.result_dir = args.result_dir

    main(cfg, use_query_mlp=not args.no_query_mlp)
