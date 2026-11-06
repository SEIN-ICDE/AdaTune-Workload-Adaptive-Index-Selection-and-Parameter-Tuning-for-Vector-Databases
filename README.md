# AdaTune: Workload-Adaptive Index Selection and Parameter Tuning for Vector Databases

AdaTune is a workload-adaptive tuning framework for [Milvus](https://milvus.io). It combines representation choice, index selection, and parameter optimization to improve search efficiency while accounting for a target Recall@10.

AdaTune minimizes mean query latency with a penalty for recall below the target. The target is a soft reference during optimization, so a returned configuration is not guaranteed to satisfy it. All recall measurements use exact nearest neighbors in the original vector space as ground truth.

## Framework

AdaTune follows five steps:

1. **Query-aware hardness analysis.** Compute query-level hardness scores from local variants of relative contrast (RC), distance concentration (DC), and local intrinsic dimensionality (LID). Aggregate the scores into mean and tail hardness statistics to select one tuning path per workload.

2. **Conditional representation refinement.** Tune ANN-friendly workloads in the original space. For ANN-hard workloads, initialize a lower-dimensional representation with PCA and refine the data vectors using a weighted residual GNN guided by an original-space k-nearest-neighbor graph. Queries use the same PCA projection without a neural correction. The evaluated refined representations have 128 dimensions.

3. **Hardness-aware tiny-probe index screening.** Sample queries across low, medium, and high hardness. Evaluate candidate index types using a fixed build configuration per type and multiple search settings, then retain promising types for full optimization.

4. **Stage 1: Index parameter optimization.** Optimize the retained index types and their build and search parameters using a cost-aware coarse-to-fine procedure. Bayesian optimization uses a Gaussian process surrogate and the UCB acquisition function to tune search parameters. Each constructed index is reused across multiple search-parameter evaluations.

5. **Stage 2: System parameter optimization.** Keep the selected index type, build parameters, and search parameters fixed, and optimize Milvus system parameters using Bayesian optimization.

The tuning path and final configuration are shared across the workload. AdaTune does not select a separate index or representation for each query.

## Optimization Objective

Both parameter optimization stages maximize:

```text
reward = -mean_latency_ms - lambda * max(0, target_recall - recall_at_10)
```

Here, `lambda` controls the penalty for a recall deficit. Tiny-probe screening uses the same reward form with its screening penalty `beta`.

QPS and p95 latency are recorded for evaluation but are not part of the reward. A configuration is considered feasible for reporting only when its measured Recall@10 meets the target.

## Repository Structure

```text
adatune/          Hardness analysis, Bayesian optimization, tuning stages,
                  and Milvus utilities
preprocess/       PCA initialization and graph-guided residual refinement
configs/          Dataset paths and parameter search spaces
run_stage1.py      Index parameter optimization
run_stage2.py      System parameter optimization
requirements.txt  Python dependencies
```

## Setup

Install the Python dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

The paper experiments use:

- Milvus 2.6.14
- PyMilvus 2.6.12
- Ubuntu 18.04.2 LTS
- Intel Core i7-6700K at 4.00 GHz
- 62 GB RAM

Before running:

- Start a Milvus standalone deployment in Docker, accessible at `localhost:19530`.
- Configure the dataset paths in `configs/*.py` and `preprocess/build_dataset.py`.
- For Stage 2, provide access to the container restart operation and the Milvus `user.yaml` configuration file.
- Ensure configuration changes take effect before benchmarking and preserve the dataset and index across container restarts.

## Datasets

The evaluation uses four workloads:

| Workload | Database vectors | Query vectors | Original dimensions | Metric |
|---|---:|---:|---:|---|
| GloVe | 1,183,514 | 10,000 | 100 | Angular |
| SIFT | 1,000,000 | 10,000 | 128 | Euclidean |
| GIST | 1,000,000 | 1,000 | 960 | Euclidean |
| Wikipedia | 100,000 | 10,000 | 768 | Cosine |

Wikipedia embeddings are constructed from the November 1, 2023 English Wikipedia snapshot (`20231101.en`) using `all-mpnet-base-v2`. The resulting 768-dimensional embeddings are normalized.

Datasets use the [ANN-Benchmarks](https://github.com/erikbern/ann-benchmarks) HDF5 layout:

- `train`: database vectors
- `test`: query vectors
- `neighbors`: original-space ground-truth neighbor IDs

For the refined path, `train` contains refined data vectors and `test` contains PCA-projected queries. **The `neighbors` field must retain the original-space ground truth.** Recomputing ground truth in the refined space would change the recall definition used in the paper.

## Usage

The commands below illustrate representation preparation and the two parameter optimization stages.

For paper reproduction, hardness analysis and tiny-probe screening must also be completed: Stage 1 uses the retained index types and probe observations from screening.

### ANN-hard example: GIST

Prepare the refined representation:

```bash
python -m preprocess.build_dataset --dataset gist
```

Run index parameter optimization:

```bash
python run_stage1.py --dataset gist --threshold 0.75
```

Run system parameter optimization using the Stage 1 result:

```bash
python run_stage2.py --dataset gist --threshold 0.75
```

### ANN-friendly example: SIFT

Use the original vectors without representation refinement:

```bash
python run_stage1.py --dataset sift --threshold 0.95
python run_stage2.py --dataset sift --threshold 0.95
```

The dataset argument is one of `sift`, `glove`, `gist`, or `wikipedia`.

The CLI accepts recall thresholds from `0.65` to `0.95` in increments of `0.05`. Accepting a threshold does not imply that the selected representation and configuration can achieve it.

## Paper Evaluation Protocol

### Candidate indexes

Tiny-probe screening considers six Milvus index types:

```text
FLAT, HNSW, IVF_FLAT, IVF_SQ8, IVF_PQ, SCANN
```

Stage 1 optimizes the types retained by screening rather than restricting every workload to a predetermined index pair.

### Tuning budgets

For each workload and target recall, the paper uses:

| Phase | Evaluations | Purpose |
|---|---:|---|
| Tiny-probe screening | 120 | 20 search settings for each of six index types |
| Stage 1 | 180 | Index type, build parameters, and search parameters |
| Stage 2 | 120 | System parameters with the index configuration fixed |

The 300 full optimization evaluations comprise Stage 1 and Stage 2. The 120 probe evaluations are additional and are excluded from the full optimization iteration count.

AdaTune's wall-clock convergence measurements include tiny-probe screening time. Initial random evaluations in Stage 2 count toward its 120-evaluation budget.

### Measurement

- One query per request (`nq = 1`)
- Top-10 retrieval
- 10 concurrent client threads
- Recall@10 against original-space ground truth
- Mean latency, p95 latency, and QPS recorded for each evaluation

Compare configurations using both the requested recall target and the achieved Recall@10. Meeting the same target does not necessarily mean that two configurations achieve the same recall.

### Recall targets and operating range

In the reported evaluation:

- GloVe and SIFT meet targets of `0.80`, `0.85`, `0.90`, and `0.95`.
- The refined GIST path meets targets of `0.70` and `0.75`.
- The refined Wikipedia path meets targets of `0.65` and `0.70`.

The current refined path does not meet the higher evaluated recall targets on GIST and Wikipedia. Its efficiency benefits should therefore be interpreted within the attained recall range.

Exact search between PCA-projected queries and refined data vectors is also evaluated against original-space ground truth. This measures neighborhood fidelity without ANN approximation error; it is distinct from the original-space parameter sweep.

## Outputs

Results are written to:

```text
results/stage{1,2}_<dataset>_thr<tau>_{best.json,all.csv}
```

Interrupted runs resume from snapshots under:

```text
results/snapshots/
```

When interpreting results, distinguish the configuration with the highest reward from configurations that satisfy the recall target. The soft recall penalty allows below-target configurations to be evaluated during tuning.
