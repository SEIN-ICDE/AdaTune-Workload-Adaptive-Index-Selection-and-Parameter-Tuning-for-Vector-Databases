AdaTune tunes a [Milvus](https://milvus.io) deployment to reach a target **Recall@10** at the lowest latency it can find.

  - **ANN-hard datasets** (GIST, Wikipedia) are first projected to 128-d by a **PCA + weighted residual GNN**. Recall is still measured against the original-space ground truth.
  - **Stage 1** uses Bayesian optimization to pick the index type (HNSW / SCANN) and tune its build and search parameters.
  - **Stage 2** keeps that index and tunes Milvus system parameters.

  Both stages maximize the same soft-constrained reward:
  
  ```
  reward = − mean_latency_ms − λ · max(0, τ − Recall@10),   λ = 1000
  ```
  
  ## Structure
  
  ```
  adatune/      hardness scoring, BO (GP-UCB), Stage 1 / Stage 2, Milvus utils
  preprocess/   PCA + GNN (+ query MLP) refinement → 128-d HDF5
  configs/      per-dataset paths and search spaces (sift, glove, gist, wikipedia)
  run_stage1.py / run_stage2.py

## Setup
  
  ```bash
  pip install -r requirements.txt
  ```
 
  You also need:
  - a Milvus standalone server running in Docker at `localhost:19530`;
  - for Stage 2, permission to restart that container and to edit its `user.yaml`.

  Datasets use the [ann-benchmarks](https://github.com/erikbern/ann-benchmarks) HDF5 format (`train`, `test`, `neighbors`). Before running, edit the paths in `configs/*.py` and in `preprocess/build_dataset.py`.
  
  ## Usage 
  
  ```bash
  # (ANN-hard only) build the 128-d dataset
  python -m preprocess.build_dataset --dataset gist

  # Stage 1: index type + build/search params
  python run_stage1.py --dataset gist --threshold 0.90

  # Stage 2: Milvus system params (uses the Stage 1 result)
  python run_stage2.py --dataset gist --threshold 0.90

- `--dataset` is one of `sift`, `glove`, `gist`, `wikipedia`.
  - `--threshold` is one of `0.65`–`0.95`, in steps of 0.05.
  - Results are written to `results/stage{1,2}_<dataset>_thr<τ>_{best.json,all.csv}`. Interrupted runs resume from `results/snapshots/`.

  ## Details

  - **Search space**
    - HNSW: `M` 4–64, `efConstruction` 40–400, `ef` 16–512.
    - SCANN: `nlist` 256–4096, `nprobe` 1–512.
  - **Stage 1** has two phases:
    - Coarse: 3 default builds per index type, with 12 search-BO iterations each.
    - Fine: for the best index type, the top coarse builds plus 2 GP-proposed builds, with 22 iterations each.
  - **Stage 2** tunes `queryNode.gracefulTime`, `rootCoord.minSegmentSizeToEnableIndex`, `queryCoord.autoHandoff` and `queryCoord.autoBalance`.
    - It runs up to 120 trials, with early stopping after 20 trials without improvement.
    - Every trial restarts Milvus.
  - **Measurement**: one query per request (`nq = 1`), top-10, 10 concurrent client threads. QPS and p95 latency are logged but are not part of the reward.

