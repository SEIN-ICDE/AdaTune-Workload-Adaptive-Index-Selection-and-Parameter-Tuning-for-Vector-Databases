"""
adatune/hardness.py

Per-query hardness scoring for AdaTune.

H(q) = RC_norm(q) + DC_norm(q) + LID_norm(q)  ∈ [0, 3]

Each metric is normalized to [0, 1] using ranges derived from
synthetic ANN-friendly and ANN-hard anchors:
  - RC  (Relative Contrast)           : larger = easier → direction flipped
  - DC  (Distance Concentration)      : larger = harder → kept as-is
  - LID (Local Intrinsic Dimensionality): larger = harder → kept as-is

Workload-level hardness combines mean H and tail H (top-p hardest queries)
to route the tuner to an ANN-friendly or ANN-hard optimization path.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Per-query metrics
# ---------------------------------------------------------------------------

def compute_rc(d_knn: np.ndarray) -> np.ndarray:
    """
    Relative Contrast.  Larger value = easier query.

    RC(q) = (mean_dist - d_1) / d_1
    where d_1 is the nearest-neighbor distance.
    """
    return (d_knn.mean(axis=1) - d_knn[:, 0]) / (d_knn[:, 0] + 1e-8)


def compute_dc(d_knn: np.ndarray) -> np.ndarray:
    """
    Distance Concentration.  Larger value = harder query.

    DC(q) = std(d) / mean(d)
    """
    return d_knn.std(axis=1) / (d_knn.mean(axis=1) + 1e-8)


def compute_lid(d_knn: np.ndarray, k: int = 20) -> np.ndarray:
    """
    Local Intrinsic Dimensionality via MLE estimator.  Larger = harder.

    Uses k-1 distances with the k-th as the radius (d_max).

    Parameters
    ----------
    d_knn : (nq, >=k) sorted ascending distances
    k     : number of neighbors (K_h in the paper)
    """
    d     = np.clip(d_knn[:, :k], 1e-12, None)
    d_max = d[:, -1:]
    log_r = np.log(d[:, :-1] / d_max)          # shape (nq, k-1)
    return -(k - 1) / log_r.sum(axis=1)


# ---------------------------------------------------------------------------
# Anchor-based normalization
# ---------------------------------------------------------------------------

@dataclass
class HardnessAnchors:
    """
    Per-metric min/max ranges calibrated from synthetic anchors.

    Friendly anchor (low hardness) supplies the min values;
    hard anchor (high hardness) supplies the max values.
    """
    rc_min:  float
    rc_max:  float
    dc_min:  float
    dc_max:  float
    lid_min: float
    lid_max: float

    @classmethod
    def from_percentiles(
        cls,
        d_knn: np.ndarray,
        lid_k: int = 20,
        low_pct: float = 5.0,
        high_pct: float = 95.0,
    ) -> "HardnessAnchors":
        """
        Estimate anchor ranges from data percentiles when pre-defined
        synthetic anchors are not available.
        """
        rc  = compute_rc(d_knn)
        dc  = compute_dc(d_knn)
        lid = compute_lid(d_knn, k=lid_k)
        return cls(
            rc_min=float(np.percentile(rc,   low_pct)),
            rc_max=float(np.percentile(rc,  high_pct)),
            dc_min=float(np.percentile(dc,   low_pct)),
            dc_max=float(np.percentile(dc,  high_pct)),
            lid_min=float(np.percentile(lid,  low_pct)),
            lid_max=float(np.percentile(lid, high_pct)),
        )


def _norm(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((x - lo) / (hi - lo + 1e-12), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Query-level hardness
# ---------------------------------------------------------------------------

def compute_query_hardness(
    d_knn: np.ndarray,
    anchors: HardnessAnchors,
    lid_k: int = 20,
) -> np.ndarray:
    """
    Compute per-query hardness H(q) ∈ [0, 3].

    Parameters
    ----------
    d_knn   : (nq, K) array of sorted ascending nearest-neighbor distances
    anchors : anchor ranges for normalization
    lid_k   : k for LID estimation

    Returns
    -------
    H : (nq,) hardness scores
    """
    rc  = compute_rc(d_knn)
    dc  = compute_dc(d_knn)
    lid = compute_lid(d_knn, k=lid_k)

    rc_norm  = 1.0 - _norm(rc,  anchors.rc_min,  anchors.rc_max)   # flip: easy → 0
    dc_norm  =       _norm(dc,  anchors.dc_min,  anchors.dc_max)
    lid_norm =       _norm(lid, anchors.lid_min, anchors.lid_max)

    return rc_norm + dc_norm + lid_norm


# ---------------------------------------------------------------------------
# Workload-level classification
# ---------------------------------------------------------------------------

@dataclass
class WorkloadProfile:
    mean_h:  float
    tail_h:  float
    is_hard: bool


def classify_workload(
    H: np.ndarray,
    mean_threshold: float = 1.5,
    tail_pct: float = 0.10,
    tail_threshold: float = 2.0,
) -> WorkloadProfile:
    """
    Classify workload as ANN-hard or ANN-friendly.

    A workload is considered hard if EITHER:
      - mean H across all queries  >= mean_threshold, OR
      - mean H of top tail_pct queries >= tail_threshold

    The ANN-hard path applies the PCA+GNN-transformed index;
    the ANN-friendly path uses the standard index without transformation.
    """
    mean_h = float(H.mean())
    n_tail = max(1, int(len(H) * tail_pct))
    tail_h = float(np.sort(H)[-n_tail:].mean())
    is_hard = (mean_h >= mean_threshold) or (tail_h >= tail_threshold)
    return WorkloadProfile(mean_h=mean_h, tail_h=tail_h, is_hard=is_hard)
