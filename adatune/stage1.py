"""
adatune/stage1.py

Stage 1: Index-type and parameter selection via Bayesian Optimization.

Flow:
  Coarse  → evaluate each default build config with COARSE_ITERS search BO
  Fine    → re-run the top-K coarse configs + GP-suggested new configs
            with FINE_ITERS search BO
  Output  → best (index_type, build_params, search_param) saved to JSON
"""

from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import pandas as pd
from pymilvus import MilvusClient

from .bo import (
    candidate_grid_norm,
    denormalize_build_params,
    fit_gp,
    log_candidates,
    normalize_build_params,
    reward,
    ucb,
)
from .milvus_utils import (
    build_collection,
    calc_recall,
    do_search,
    drop_if_exists,
    load_dataset,
)

SEED              = 42
PENALTY_LAMBDA    = 1000.0
COARSE_ITERS      = 12
FINE_ITERS        = 22
FINE_TOP_K        = 4
FINE_N_EXPLORE    = 2
UCB_KAPPA_SEARCH  = 2.0
UCB_KAPPA_EXPLORE = 3.0


# ---------------------------------------------------------------------------
# Search-parameter BO (inner loop)
# ---------------------------------------------------------------------------

def _run_search_bo(
    client: MilvusClient,
    col_name: str,
    index_type: str,
    xq: np.ndarray,
    gt: np.ndarray,
    search_cfg: dict,
    n_iters: int,
    threshold: float,
    global_counter: list[int],
    snapshot_path: str | None,
) -> tuple[list[dict], int, float]:
    """
    Run BO over the search parameter (ef for HNSW, nprobe for SCANN).
    Returns (records, best_search_val, best_reward).
    """
    search_col = list(search_cfg.keys())[0]
    lo, hi     = search_cfg[search_col]["range"]
    cands      = log_candidates(lo, hi)

    X, y     = np.empty((0, 1)), np.empty(0)
    records  = []
    done     = 0

    if snapshot_path and os.path.exists(snapshot_path):
        try:
            df = pd.read_csv(snapshot_path, on_bad_lines="skip")
            if len(df) > 0:
                X = df[search_col].values.reshape(-1, 1).astype(float)
                y = df["reward"].values.astype(float)
                records      = df.to_dict("records")
                global_counter[0] = int(df["global_iter"].max())
                done         = len(df)
                print(f"      ↩ snapshot: {done} iters loaded")
        except Exception as e:
            print(f"      snapshot load failed: {e}")

    rng = np.random.default_rng(SEED + global_counter[0])

    for i in range(done, n_iters):
        if len(y) < 2:
            val = int(round(math.exp(rng.uniform(math.log(max(lo, 1)), math.log(hi)))))
        else:
            scores = ucb(fit_gp(X, y), cands, UCB_KAPPA_SEARCH)
            val    = int(round(cands[np.argmax(scores), 0]))
        val = int(np.clip(val, lo, hi))

        global_counter[0] += 1
        g_iter = global_counter[0]

        results, lat, p95, qps = do_search(client, col_name, index_type, xq, val)
        rec    = calc_recall(results, gt)
        rew    = reward(rec, lat, threshold, PENALTY_LAMBDA)

        x_new = np.array([[float(val)]])
        X     = np.vstack([X, x_new]) if len(X) else x_new
        y     = np.append(y, rew)

        row = {
            search_col: val, "recall": rec, "latency_ms": lat,
            "latency_p95ms": p95, "qps": qps, "reward": rew,
            "iter": i + 1, "global_iter": g_iter, "timestamp": time.time(),
        }
        records.append(row)
        if snapshot_path:
            pd.DataFrame([row]).to_csv(
                snapshot_path, mode="a",
                header=not os.path.exists(snapshot_path), index=False,
            )
        print(f"      iter {i+1:02d} [g={g_iter:03d}]  {search_col}={val:<5}  "
              f"recall={rec:.4f}  lat={lat:.2f}ms  reward={rew:.3f}")

    if len(y) == 0:
        return records, int(cands[0, 0]), -float("inf")
    best = int(np.argmax(y))
    return records, int(round(X[best, 0])), float(y[best])


# ---------------------------------------------------------------------------
# Exploration build params (GP over build space)
# ---------------------------------------------------------------------------

def _explore_build_params(
    build_results: list[dict],
    build_cfg: dict,
    n_explore: int,
) -> list[dict]:
    if len(build_results) < 2:
        rng = np.random.default_rng(SEED + 99)
        return [
            {
                c: int(round(math.exp(rng.uniform(math.log(cfg["range"][0]),
                                                   math.log(cfg["range"][1])))))
                   if cfg["scale"] == "log" else int(round(rng.uniform(*cfg["range"])))
                for c, cfg in build_cfg.items()
            }
            for _ in range(n_explore)
        ]

    Xn    = normalize_build_params([r["build_params"] for r in build_results], build_cfg)
    y_vals = np.array([r["best_reward"] for r in build_results])
    gp    = fit_gp(Xn, y_vals)
    cands = candidate_grid_norm(build_cfg, n_each=20)
    scores = ucb(gp, cands, UCB_KAPPA_EXPLORE)

    cols     = list(build_cfg.keys())
    existing = {tuple(r["build_params"][c] for c in cols) for r in build_results}
    explored = []
    for idx in np.argsort(scores)[::-1]:
        bp  = denormalize_build_params(cands[idx], build_cfg)
        key = tuple(bp[c] for c in cols)
        if key not in existing:
            explored.append(bp)
            existing.add(key)
        if len(explored) >= n_explore:
            break
    return explored


# ---------------------------------------------------------------------------
# Coarse / Fine phases
# ---------------------------------------------------------------------------

def _coarse(client, index_type, dim, xq, gt, cfg, global_counter, threshold,
            result_dir, snapshot_dir):
    search_cfg   = cfg["index"][index_type]["search"]
    search_col   = list(search_cfg.keys())[0]
    default_grid = cfg["index"][index_type]["build_default_grid"]
    all_recs, build_results = [], []
    tag = str(threshold).replace(".", "")

    print(f"\n  [{index_type}] Coarse ({COARSE_ITERS} iters/build)")
    for i, bp in enumerate(default_grid):
        col_name  = f"adatune_{cfg['name']}_thr{tag}_c_{index_type}_{i}".lower()
        snap_path = os.path.join(snapshot_dir, f"snap_{cfg['name']}_thr{tag}_coarse_{index_type}_b{i+1}.csv")
        print(f"\n    build {i+1}/{len(default_grid)}: {bp}")

        if os.path.exists(snap_path):
            try:
                df = pd.read_csv(snap_path, on_bad_lines="skip")
                if len(df) >= COARSE_ITERS:
                    global_counter[0] = max(global_counter[0], int(df["global_iter"].max()))
                    best_idx = df["reward"].idxmax()
                    build_results.append({
                        "build_params": bp,
                        "best_search_param": int(df.loc[best_idx, search_col]),
                        "best_reward": float(df.loc[best_idx, "reward"]),
                    })
                    all_recs.extend(df.to_dict("records"))
                    print(f"    ↩ cached, skipping")
                    continue
            except Exception:
                pass

        try:
            build_collection(client, col_name, dim, index_type, bp,
                             cfg["ds_path"], cfg["metric"])
        except Exception as e:
            print(f"    build failed: {e}")
            drop_if_exists(client, col_name)
            continue

        hi = min(search_cfg[search_col]["range"][1],
                 bp.get("nlist", search_cfg[search_col]["range"][1])) \
             if index_type == "SCANN" else search_cfg[search_col]["range"][1]
        sc = {search_col: {**search_cfg[search_col], "range": (search_cfg[search_col]["range"][0], hi)}}

        recs, best_sp, best_rew = _run_search_bo(
            client, col_name, index_type, xq, gt, sc,
            COARSE_ITERS, threshold, global_counter, snap_path,
        )
        all_recs.extend(recs)
        build_results.append({"build_params": bp, "best_search_param": best_sp,
                               "best_reward": best_rew})
        drop_if_exists(client, col_name)

    if not build_results:
        return None, []
    best = max(build_results, key=lambda r: r["best_reward"])
    return {
        "index_type": index_type,
        "best_build_params": best["build_params"],
        "best_search_param": best["best_search_param"],
        "best_reward": best["best_reward"],
        "all_build_results": build_results,
    }, all_recs


def _fine(client, index_type, dim, xq, gt, cfg, coarse_result,
          global_counter, threshold, snapshot_dir):
    search_cfg = cfg["index"][index_type]["search"]
    search_col = list(search_cfg.keys())[0]
    tag        = str(threshold).replace(".", "")

    top_k = [r["build_params"] for r in
              sorted(coarse_result["all_build_results"],
                     key=lambda r: r["best_reward"], reverse=True)[:FINE_TOP_K]]
    explore   = _explore_build_params(
        coarse_result["all_build_results"],
        cfg["index"][index_type]["build"],
        FINE_N_EXPLORE,
    )
    candidates = top_k + explore
    all_recs, fine_results = [], []

    print(f"\n  [{index_type}] Fine ({FINE_ITERS} iters/build, {len(candidates)} builds)")
    for i, bp in enumerate(candidates):
        col_name  = f"adatune_{cfg['name']}_thr{tag}_f_{index_type}_{i}".lower()
        snap_path = os.path.join(snapshot_dir, f"snap_{cfg['name']}_thr{tag}_fine_{index_type}_b{i+1}.csv")
        print(f"\n  build {i+1}/{len(candidates)}: {bp}")

        if os.path.exists(snap_path):
            try:
                df = pd.read_csv(snap_path, on_bad_lines="skip")
                if len(df) >= FINE_ITERS:
                    global_counter[0] = max(global_counter[0], int(df["global_iter"].max()))
                    best_idx = df["reward"].idxmax()
                    fine_results.append({
                        "build_params": bp,
                        "best_search_param": int(df.loc[best_idx, search_col]),
                        "best_reward": float(df.loc[best_idx, "reward"]),
                    })
                    all_recs.extend(df.to_dict("records"))
                    print(f"  ↩ cached, skipping")
                    continue
            except Exception:
                pass

        try:
            build_collection(client, col_name, dim, index_type, bp,
                             cfg["ds_path"], cfg["metric"])
        except Exception as e:
            print(f"  build failed: {e}")
            drop_if_exists(client, col_name)
            continue

        hi = min(search_cfg[search_col]["range"][1],
                 bp.get("nlist", search_cfg[search_col]["range"][1])) \
             if index_type == "SCANN" else search_cfg[search_col]["range"][1]
        sc = {search_col: {**search_cfg[search_col], "range": (search_cfg[search_col]["range"][0], hi)}}

        recs, best_sp, best_rew = _run_search_bo(
            client, col_name, index_type, xq, gt, sc,
            FINE_ITERS, threshold, global_counter, snap_path,
        )
        all_recs.extend(recs)
        fine_results.append({"build_params": bp, "best_search_param": best_sp,
                              "best_reward": best_rew})
        drop_if_exists(client, col_name)

    if not fine_results:
        return None, []
    best = max(fine_results, key=lambda r: r["best_reward"])
    return {
        "index_type": index_type,
        "best_build_params": best["build_params"],
        "best_search_param": best["best_search_param"],
        "best_reward": best["best_reward"],
    }, all_recs


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_stage1(cfg: dict, threshold: float, milvus_uri: str = "http://localhost:19530") -> dict:
    """
    Run Stage 1 for a given dataset config and recall threshold.

    Parameters
    ----------
    cfg       : dataset config dict (see configs/)
    threshold : recall threshold (e.g. 0.90)
    milvus_uri: Milvus server URI

    Returns
    -------
    best_info dict, also saved to results/stage1_<name>_thr<tag>_best.json
    """
    tag        = str(threshold).replace(".", "")
    result_dir = cfg.get("result_dir", "results")
    snap_dir   = os.path.join(result_dir, "snapshots")
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(snap_dir,   exist_ok=True)

    print("=" * 70)
    print(f"  Stage 1  dataset={cfg['name']}  threshold={threshold}")
    print(f"  coarse={COARSE_ITERS}/build  fine={FINE_ITERS}/build")
    print("=" * 70)

    client         = MilvusClient(milvus_uri)
    dim, xq, gt   = load_dataset(cfg["ds_path"])
    global_counter = [0]
    all_records    = []

    # --- Coarse ---
    coarse_results = {}
    for itype in cfg["index_types"]:
        res, recs = _coarse(client, itype, dim, xq, gt, cfg,
                            global_counter, threshold, result_dir, snap_dir)
        if res is not None:
            coarse_results[itype] = res
            all_records.extend(recs)

    if not coarse_results:
        raise RuntimeError("All coarse builds failed.")

    selected = max(coarse_results, key=lambda t: coarse_results[t]["best_reward"])
    print(f"\n  ★ Selected index: {selected}  (reward={coarse_results[selected]['best_reward']:.4f})")

    # --- Fine ---
    fine_result, fine_recs = _fine(
        client, selected, dim, xq, gt, cfg,
        coarse_results[selected], global_counter, threshold, snap_dir,
    )
    all_records.extend(fine_recs)

    if fine_result is None:
        fine_result = {k: coarse_results[selected][k]
                       for k in ("index_type", "best_build_params",
                                 "best_search_param", "best_reward")}

    # --- Save ---
    df_all   = pd.DataFrame(all_records)
    all_csv  = os.path.join(result_dir, f"stage1_{cfg['name']}_thr{tag}_all.csv")
    df_all.to_csv(all_csv, index=False)

    search_col   = list(cfg["index"][fine_result["index_type"]]["search"].keys())[0]
    build_cols   = list(cfg["index"][fine_result["index_type"]]["build"].keys())
    best_row     = df_all.loc[df_all["reward"].idxmax()]
    build_params = {c: int(best_row[c]) for c in build_cols
                    if c in best_row and pd.notna(best_row[c])}

    best_info = {
        "dataset":        cfg["name"],
        "threshold":      threshold,
        "penalty_lambda": PENALTY_LAMBDA,
        "index_type":     fine_result["index_type"],
        "build_params":   build_params,
        "search_val":     int(best_row.get(search_col, fine_result["best_search_param"])),
        "best_reward":    float(best_row["reward"]),
        "recall":         float(best_row["recall"]),
        "latency_ms":     float(best_row["latency_ms"]),
        "qps":            float(best_row["qps"]),
        "total_iters":    int(global_counter[0]),
    }

    json_path = os.path.join(result_dir, f"stage1_{cfg['name']}_thr{tag}_best.json")
    with open(json_path, "w") as f:
        json.dump(best_info, f, indent=2)

    print(f"\n  [Stage 1 result]  recall={best_info['recall']:.4f}  "
          f"QPS={best_info['qps']:.0f}  total_iters={global_counter[0]}")
    print(f"  saved: {json_path}")
    return best_info
