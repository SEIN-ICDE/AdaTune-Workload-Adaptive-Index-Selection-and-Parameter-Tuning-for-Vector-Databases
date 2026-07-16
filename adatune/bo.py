"""
adatune/bo.py

Bayesian Optimization utilities shared by Stage 1 and Stage 2.
Uses Gaussian Process regression with UCB acquisition.
"""

from __future__ import annotations

import math
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

SEED = 42


# ---------------------------------------------------------------------------
# GP helpers
# ---------------------------------------------------------------------------

def fit_gp(X: np.ndarray, y: np.ndarray) -> GaussianProcessRegressor:
    mask = np.isfinite(y)
    X, y = X[mask], y[mask]
    kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(1e-3)
    gp = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=5,
        normalize_y=True, random_state=SEED,
    )
    gp.fit(X, y)
    return gp


def ucb(gp: GaussianProcessRegressor, candidates: np.ndarray, kappa: float) -> np.ndarray:
    mu, sigma = gp.predict(candidates, return_std=True)
    return mu + kappa * sigma


# ---------------------------------------------------------------------------
# Parameter space helpers
# ---------------------------------------------------------------------------

def log_candidates(lo: int, hi: int, n: int = 100) -> np.ndarray:
    """Log-spaced integer candidates in [lo, hi]."""
    pts = np.unique(
        np.clip(
            np.round(np.exp(np.linspace(math.log(max(lo, 1)), math.log(hi), n))).astype(int),
            lo, hi,
        )
    )
    return pts.reshape(-1, 1).astype(float)


def normalize_param(val: float, lo: float, hi: float, scale: str) -> float:
    if scale == "log":
        return (math.log(val) - math.log(lo)) / (math.log(hi) - math.log(lo))
    return (val - lo) / (hi - lo)


def denormalize_param(norm_val: float, lo: float, hi: float, scale: str) -> float:
    if scale == "log":
        return math.exp(norm_val * (math.log(hi) - math.log(lo)) + math.log(lo))
    return norm_val * (hi - lo) + lo


def normalize_build_params(
    params_list: list[dict],
    build_cfg: dict,
) -> np.ndarray:
    """Normalize a list of build-param dicts into a 2-D float array."""
    cols = list(build_cfg.keys())
    X = np.array([[p[c] for c in cols] for p in params_list], dtype=float)
    Xn = np.zeros_like(X)
    for j, c in enumerate(cols):
        cfg = build_cfg[c]
        lo, hi = cfg["range"]
        Xn[:, j] = normalize_param(X[:, j], lo, hi, cfg["scale"])
    return Xn


def denormalize_build_params(xn: np.ndarray, build_cfg: dict) -> dict:
    """Denormalize a single normalized build-param vector into a dict."""
    cols = list(build_cfg.keys())
    result = {}
    for j, c in enumerate(cols):
        cfg = build_cfg[c]
        lo, hi = cfg["range"]
        val = denormalize_param(float(xn[j]), lo, hi, cfg["scale"])
        result[c] = int(np.clip(round(val), lo, hi))
    return result


def candidate_grid_norm(build_cfg: dict, n_each: int = 20) -> np.ndarray:
    """Build a normalized candidate grid over all build parameters."""
    cols  = list(build_cfg.keys())
    grids = [np.linspace(0.0, 1.0, n_each) for _ in cols]
    if len(grids) == 1:
        return grids[0].reshape(-1, 1)
    mesh = np.meshgrid(*grids, indexing="ij")
    return np.stack([m.ravel() for m in mesh], axis=1)


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

def reward(recall: float, latency_ms: float, threshold: float, penalty: float = 1000.0) -> float:
    """Penalized reward: maximize QPS (proxy: -latency) subject to recall >= threshold."""
    return -latency_ms - penalty * max(0.0, threshold - recall)
