"""
adatune/stage2.py

Stage 2: System-parameter tuning via Bayesian Optimization.

Uses the best index/search params from Stage 1 (fixed collection),
then optimizes Milvus system parameters by restarting the server
and measuring recall+QPS under each configuration.

System parameters tuned:
  - queryNode.gracefulTime
  - rootCoord.minSegmentSizeToEnableIndex
  - queryCoord.autoHandoff
  - queryCoord.autoBalance
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import time

import numpy as np
import pandas as pd
import requests
import yaml
from concurrent.futures import ThreadPoolExecutor
from pymilvus import MilvusClient, DataType

from .bo import fit_gp, log_candidates, normalize_param, denormalize_param, reward, ucb
from .milvus_utils import calc_recall, make_search_params

SEED           = 42
PENALTY_LAMBDA = 1000.0

S2_N_ITER        = 120
S2_N_RANDOM_INIT = 5
S2_REBUILD_EVERY = 20
S2_PATIENCE      = 20

UCB_KAPPA = 2.0
K         = 10
N_WORKERS = 10

SYS_PARAMS = [
    {
        "name": "queryNode.gracefulTime",
        "key":  ("queryNode", "gracefulTime"),
        "type": "int", "min": 0, "max": 600_000, "scale": "linear", "default": 5_000,
    },
    {
        "name": "rootCoord.minSegmentSizeToEnableIndex",
        "key":  ("rootCoord", "minSegmentSizeToEnableIndex"),
        "type": "int", "min": 1_024, "max": 65_536, "scale": "log", "default": 1_024,
    },
    {
        "name": "queryCoord.autoHandoff",
        "key":  ("queryCoord", "autoHandoff"),
        "type": "bool", "min": 0, "max": 1, "scale": "linear", "default": 1,
    },
    {
        "name": "queryCoord.autoBalance",
        "key":  ("queryCoord", "autoBalance"),
        "type": "bool", "min": 0, "max": 1, "scale": "linear", "default": 1,
    },
]
N_SYS = len(SYS_PARAMS)


# ---------------------------------------------------------------------------
# System-param helpers
# ---------------------------------------------------------------------------

def _norm_sys(val: float, p: dict) -> float:
    return normalize_param(val, p["min"], p["max"], p["scale"])


def _denorm_sys(norm: float, p: dict) -> int | bool:
    raw = denormalize_param(norm, p["min"], p["max"], p["scale"])
    if p["type"] == "bool":
        return int(round(raw))
    return int(np.clip(round(raw), p["min"], p["max"]))


def _norm_sys_vec(vals: list) -> np.ndarray:
    return np.array([_norm_sys(v, SYS_PARAMS[j]) for j, v in enumerate(vals)])


def _to_sys_dict(norm_vec: np.ndarray) -> dict:
    return {p["name"]: _denorm_sys(norm_vec[j], p)
            for j, p in enumerate(SYS_PARAMS)}


def _write_user_yaml(sys_dict: dict, yaml_path: str) -> None:
    cfg: dict = {}
    for name, val in sys_dict.items():
        keys = name.split(".")
        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = val
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f)


def _restore_defaults(yaml_path: str) -> None:
    _write_user_yaml({p["name"]: p["default"] for p in SYS_PARAMS}, yaml_path)


# ---------------------------------------------------------------------------
# Milvus restart helpers
# ---------------------------------------------------------------------------

def _restart_milvus(container: str) -> None:
    for cmd in [f"docker stop {container}", f"docker start {container}"]:
        r = subprocess.run(["sg", "docker", "-c", cmd],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"{cmd} failed: {r.stderr}")
        if "stop" in cmd:
            time.sleep(10)


def _wait_ready(health_url: str, milvus_uri: str, timeout: int = 120) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(health_url, timeout=2).status_code == 200:
                break
        except Exception:
            pass
        time.sleep(2)
    else:
        raise TimeoutError("Milvus not healthy after restart")
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            MilvusClient(milvus_uri).list_collections()
            return
        except Exception:
            time.sleep(2)
    raise TimeoutError("Milvus client connection failed after restart")


def _restart_and_wait(cfg: dict) -> float:
    t0 = time.time()
    _restart_milvus(cfg["container_name"])
    time.sleep(3)
    _wait_ready(cfg["health_url"], cfg["milvus_uri"])
    time.sleep(60)
    return time.time() - t0


# ---------------------------------------------------------------------------
# Collection setup / benchmark
# ---------------------------------------------------------------------------

def _setup_collection(client: MilvusClient, col_name: str, xb: np.ndarray,
                      dim: int, index_type: str, build_params: dict,
                      metric: str) -> None:
    if client.has_collection(col_name):
        state = client.get_load_state(collection_name=col_name)
        if state.get("state") != "Loaded":
            client.load_collection(col_name)
        return

    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id",     DataType.INT64,        is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
    client.create_collection(collection_name=col_name, schema=schema)

    for start in range(0, len(xb), 10_000):
        end  = min(start + 10_000, len(xb))
        data = [{"id": i, "vector": xb[i].tolist()} for i in range(start, end)]
        client.insert(collection_name=col_name, data=data)
    client.flush(col_name)

    mp = dict(build_params)
    if index_type == "SCANN":
        mp["with_raw_data"] = True
    ip = client.prepare_index_params()
    ip.add_index(field_name="vector", index_type=index_type,
                 metric_type=metric, params=mp)
    client.create_index(col_name, ip)
    client.load_collection(col_name)


def _benchmark(client: MilvusClient, col_name: str, sp: dict,
               xq: np.ndarray, gt: np.ndarray) -> tuple[float, float, float, float]:
    def _q(q):
        t0  = time.perf_counter()
        res = client.search(collection_name=col_name, data=[q.tolist()],
                            limit=K, search_params=sp, output_fields=[])
        return res[0], (time.perf_counter() - t0) * 1000

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        list(ex.map(_q, xq[:200]))          # warmup

    t_s = time.perf_counter()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        outcomes = list(ex.map(_q, xq))
    total = time.perf_counter() - t_s

    results = [r for r, _ in outcomes]
    lats    = [l for _, l in outcomes]
    hits    = sum(len({r["id"] for r in res} & set(gt[i].tolist()))
                  for i, res in enumerate(results))
    rec     = hits / (len(results) * K)
    return rec, float(np.mean(lats)), float(np.percentile(lats, 95)), len(xq) / total


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_stage2(cfg: dict, threshold: float) -> dict:
    """
    Run Stage 2 for a given dataset config and recall threshold.

    Requires Stage 1 JSON to exist at:
        results/stage1_<name>_thr<tag>_best.json

    Returns best_info dict, also saved to results/stage2_<name>_thr<tag>_best.json
    """
    tag        = str(threshold).replace(".", "")
    result_dir = cfg.get("result_dir", "results")
    snap_dir   = os.path.join(result_dir, "snapshots")
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(snap_dir,   exist_ok=True)

    # Load Stage 1 result
    s1_path = os.path.join(result_dir, f"stage1_{cfg['name']}_thr{tag}_best.json")
    if not os.path.exists(s1_path):
        raise FileNotFoundError(f"Stage 1 result not found: {s1_path}")
    with open(s1_path) as f:
        s1 = json.load(f)

    index_type   = s1["index_type"]
    build_params = s1["build_params"]
    search_val   = s1["search_val"]
    sp           = make_search_params(index_type, search_val)

    print("=" * 70)
    print(f"  Stage 2  dataset={cfg['name']}  threshold={threshold}")
    print(f"  index={index_type}  build={build_params}  search_val={search_val}")
    print("=" * 70)

    # Load data
    import h5py
    with h5py.File(cfg["ds_path"], "r") as f:
        xb  = f["train"][:].astype("float32")
        xq  = f["test"][:].astype("float32")
        gt  = f["neighbors"][:, :K].astype(np.int64)
    dim = xb.shape[1]

    col_name  = f"adatune_{cfg['name']}_thr{tag}_s2".lower()
    yaml_path = cfg.get("user_yaml_path", "/home/sein/user.yaml")
    snap_path = os.path.join(snap_dir, f"snap_{cfg['name']}_thr{tag}_s2.csv")

    # Restore defaults → restart → build collection once
    _restore_defaults(yaml_path)
    _restart_and_wait(cfg)
    client = MilvusClient(cfg["milvus_uri"])
    _setup_collection(client, col_name, xb, dim, index_type, build_params, cfg["metric"])

    # BO over system parameters
    rng        = np.random.default_rng(SEED)
    X, y       = np.empty((0, N_SYS)), np.empty(0)
    records    = []
    best_rew   = -float("inf")
    no_improve = 0

    # Resume from snapshot
    done = 0
    if os.path.exists(snap_path):
        try:
            df_snap = pd.read_csv(snap_path, on_bad_lines="skip")
            if len(df_snap) > 0:
                param_names = [p["name"] for p in SYS_PARAMS]
                Xn = np.array([[_norm_sys(row[p], SYS_PARAMS[j])
                                for j, p in enumerate(param_names)]
                               for _, row in df_snap.iterrows()])
                yn = df_snap["reward"].values.astype(float)
                X, y   = Xn, yn
                records = df_snap.to_dict("records")
                done    = len(df_snap)
                best_rew = float(y.max())
                print(f"  ↩ snapshot: {done} iters loaded")
        except Exception as e:
            print(f"  snapshot load failed: {e}")

    for i in range(done, S2_N_ITER):
        # Sample system params
        if i < S2_N_RANDOM_INIT or len(y) < 2:
            norm_vec = rng.uniform(0.0, 1.0, N_SYS)
        else:
            cands  = rng.uniform(0.0, 1.0, (500, N_SYS))
            scores = ucb(fit_gp(X, y), cands, UCB_KAPPA)
            norm_vec = cands[np.argmax(scores)]

        sys_dict = _to_sys_dict(norm_vec)
        print(f"\n  iter {i+1:03d}  params={sys_dict}")

        # Apply → restart → evaluate
        _write_user_yaml(sys_dict, yaml_path)
        try:
            _restart_and_wait(cfg)
            client = MilvusClient(cfg["milvus_uri"])

            # Rebuild collection every S2_REBUILD_EVERY iters to prevent recall drift
            if (i + 1) % S2_REBUILD_EVERY == 0:
                _restore_defaults(yaml_path)
                _restart_and_wait(cfg)
                client = MilvusClient(cfg["milvus_uri"])
                _write_user_yaml(sys_dict, yaml_path)
                _restart_and_wait(cfg)
                client = MilvusClient(cfg["milvus_uri"])
                _setup_collection(client, col_name, xb, dim, index_type, build_params, cfg["metric"])

            rec, lat, p95, qps = _benchmark(client, col_name, sp, xq, gt)
        except Exception as e:
            print(f"  eval failed: {e}")
            _restore_defaults(yaml_path)
            _restart_and_wait(cfg)
            client = MilvusClient(cfg["milvus_uri"])
            continue

        rew = reward(rec, lat, threshold, PENALTY_LAMBDA)
        print(f"  recall={rec:.4f}  lat={lat:.2f}ms  qps={qps:.0f}  reward={rew:.3f}")

        x_new  = norm_vec.reshape(1, -1)
        X      = np.vstack([X, x_new]) if len(X) else x_new
        y      = np.append(y, rew)

        row = {p["name"]: sys_dict[p["name"]] for p in SYS_PARAMS}
        row.update({"recall": rec, "latency_ms": lat, "latency_p95ms": p95,
                    "qps": qps, "reward": rew, "iter": i + 1, "timestamp": time.time()})
        records.append(row)
        pd.DataFrame([row]).to_csv(
            snap_path, mode="a",
            header=not os.path.exists(snap_path), index=False,
        )

        if rew > best_rew:
            best_rew   = rew
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= S2_PATIENCE:
                print(f"  early stop: no improvement for {S2_PATIENCE} iters")
                break

    # Restore defaults after tuning
    _restore_defaults(yaml_path)
    _restart_and_wait(cfg)

    df_all  = pd.DataFrame(records)
    all_csv = os.path.join(result_dir, f"stage2_{cfg['name']}_thr{tag}_all.csv")
    df_all.to_csv(all_csv, index=False)

    best_row  = df_all.loc[df_all["reward"].idxmax()]
    best_info = {
        "dataset":       cfg["name"],
        "threshold":     threshold,
        "index_type":    index_type,
        "build_params":  build_params,
        "search_val":    search_val,
        "sys_params":    {p["name"]: int(best_row[p["name"]]) for p in SYS_PARAMS},
        "best_reward":   float(best_row["reward"]),
        "recall":        float(best_row["recall"]),
        "latency_ms":    float(best_row["latency_ms"]),
        "qps":           float(best_row["qps"]),
        "total_iters":   len(records),
    }

    json_path = os.path.join(result_dir, f"stage2_{cfg['name']}_thr{tag}_best.json")
    with open(json_path, "w") as f:
        json.dump(best_info, f, indent=2)

    print(f"\n  [Stage 2 result]  recall={best_info['recall']:.4f}  "
          f"QPS={best_info['qps']:.0f}  total_iters={len(records)}")
    print(f"  saved: {json_path}")
    return best_info
