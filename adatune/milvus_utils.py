"""
adatune/milvus_utils.py

Milvus collection management and search utilities.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
from pymilvus import MilvusClient, DataType

INSERT_BATCH = 10_000
N_WORKERS    = 10
K            = 10


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_dataset(ds_path: str, k: int = K) -> tuple[int, np.ndarray, np.ndarray]:
    """Returns (dim, xq, gt)."""
    with h5py.File(ds_path, "r") as f:
        dim = f["train"].shape[1]
        xq  = f["test"][:].astype("float32")
        gt  = f["neighbors"][:, :k].astype(np.int64)
    return dim, xq, gt


# ---------------------------------------------------------------------------
# Collection lifecycle
# ---------------------------------------------------------------------------

def drop_if_exists(client: MilvusClient, col_name: str) -> None:
    if client.has_collection(col_name):
        client.drop_collection(col_name)


def build_collection(
    client: MilvusClient,
    col_name: str,
    dim: int,
    index_type: str,
    build_params: dict,
    ds_path: str,
    metric: str,
) -> None:
    drop_if_exists(client, col_name)

    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id",     DataType.INT64,        is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
    client.create_collection(collection_name=col_name, schema=schema)

    with h5py.File(ds_path, "r") as f:
        n = f["train"].shape[0]
        for start in range(0, n, INSERT_BATCH):
            end   = min(start + INSERT_BATCH, n)
            batch = f["train"][start:end].astype("float32")
            data  = [{"id": i, "vector": batch[i - start].tolist()} for i in range(start, end)]
            client.insert(collection_name=col_name, data=data)
            print(f"    insert {end}/{n}", end="\r")
    client.flush(col_name)
    print(f"    inserted {n} vectors")

    milvus_params = dict(build_params)
    if index_type == "SCANN":
        milvus_params["with_raw_data"] = True

    idx_params = client.prepare_index_params()
    idx_params.add_index(
        field_name="vector", index_type=index_type,
        metric_type=metric, params=milvus_params,
    )
    client.create_index(col_name, idx_params)
    client.load_collection(col_name)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def make_search_params(index_type: str, search_val: int, k: int = K) -> dict:
    if index_type == "HNSW":
        return {"params": {"ef": int(search_val)}}
    nprobe = int(search_val)
    return {"params": {"nprobe": nprobe, "reorder_k": max(nprobe * 5, k * 2)}}


def do_search(
    client: MilvusClient,
    col_name: str,
    index_type: str,
    xq: np.ndarray,
    search_val: int,
    k: int = K,
) -> tuple[list, float, float, float]:
    """Returns (results, mean_lat_ms, p95_lat_ms, qps)."""
    sp = make_search_params(index_type, search_val, k)

    def _query(q):
        t0  = time.perf_counter()
        res = client.search(
            collection_name=col_name, data=[q.tolist()],
            limit=k, search_params=sp, output_fields=[],
        )
        return res[0], (time.perf_counter() - t0) * 1000

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        outcomes = list(ex.map(_query, xq))

    lats    = [lat for _, lat in outcomes]
    results = [res for res, _ in outcomes]
    qps     = len(xq) / (time.perf_counter() - t_start)
    return results, float(np.mean(lats)), float(np.percentile(lats, 95)), qps


def calc_recall(results: list, gt: np.ndarray, k: int = K) -> float:
    hits = sum(
        len({r["id"] for r in res} & set(gt[i].tolist()))
        for i, res in enumerate(results)
    )
    return hits / (len(results) * k)
