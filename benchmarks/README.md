# MNIST End-to-End Benchmark

Runs distributed MNIST training through the full O.N.O stack (Docker → workers → parameter server / ring AllReduce → orchestrator), measures accuracy and training time, and generates comparison plots.

## Setup

```bash
.venv/bin/python -c "import orchestra, torch, matplotlib; print('OK')"
```

If you don't have the venv:

```bash
cd orchestra-py && maturin develop --release && cd ..
pip install torch safetensors matplotlib
```

`/etc/hosts` must contain entries for `worker-0..N` and `server-0..N`. The benchmark exits with copy-pasteable fix instructions if any are missing.

```bash
sudo bash -c "echo '127.0.0.1 worker-0\n127.0.0.1 worker-1\n127.0.0.1 worker-2\n127.0.0.1 server-0\n127.0.0.1 server-1' >> /etc/hosts"
```

## Running

```bash
# All 22 combinations (~55–60 min)
.venv/bin/python benchmarks/mnist_e2e.py

# Specific numbered runs (only updates those bars in the plots)
.venv/bin/python benchmarks/mnist_e2e.py --runs 1 11 22

# Keep containers after run / force image rebuild
.venv/bin/python benchmarks/mnist_e2e.py --keep-containers --rebuild
```

Partial runs only regenerate comparison plots that include at least one of the executed runs. All other plots keep their last historical values from previous JSONL files.

## Run Table

| # | Model | Algorithm | Workers | Servers | Serializer | Sync | Max Epochs |
|---|-------|-----------|---------|---------|------------|------|------------|
| 1 | dense_small | PS | 2 | 1 | base | barrier | 80 |
| 2 | dense_large | PS | 2 | 1 | base | barrier | 120 |
| 3 | dense_small | PS | 3 | 1 | base | barrier | 150 |
| 4 | dense_large | PS | 3 | 1 | base | barrier | 200 |
| 5 | dense_small | PS | 3 | 2 | base | barrier | 150 |
| 6 | dense_large | PS | 3 | 2 | base | barrier | 200 |
| 7 | dense_small | PS | 2 | 1 | sparse | barrier | 80 |
| 8 | dense_large | PS | 2 | 1 | sparse | barrier | 120 |
| 9 | dense_small | PS | 2 | 1 | base | nonblocking | 80 |
| 10 | dense_large | PS | 2 | 1 | base | nonblocking | 120 |
| 11 | dense_small | AllReduce | 2 | 0 | base | — | 150 |
| 12 | dense_large | AllReduce | 2 | 0 | base | — | 150 |
| 13 | dense_small | AllReduce | 3 | 0 | base | — | 150 |
| 14 | dense_large | AllReduce | 3 | 0 | base | — | 200 |
| 16 | dense_small | AllReduce | 2 | 0 | sparse | — | 150 |
| 17 | dense_large | AllReduce | 2 | 0 | sparse | — | 150 |
| 19 | dense_small | AllReduce | 3 | 0 | sparse | — | 150 |
| 20 | dense_large | AllReduce | 3 | 0 | sparse | — | 200 |
| 22 | conv_small_softmax | AllReduce | 2 | 0 | base | — | 100 |
| 23 | conv_small_softmax | AllReduce | 3 | 0 | base | — | 100 |
| 30 | conv_small_softmax | PS | 2 | 1 | base | barrier | 100 |
| 32 | conv_small_softmax | PS | 3 | 1 | base | barrier | 250 |

**Not included:** `conv_large_softmax` (Conv2d 32 filters) and conv sparse runs.
`conv_large` requires ~15s/epoch making a 200-epoch run ~50 min; the sparse serializer
adds a 6–7× per-epoch overhead for conv tensors. Both dimensions are covered by dense
runs (large models) and dense sparse runs (sparse gradient compression) respectively.
`conv_small_softmax` is sufficient to validate conv support end-to-end.

## Results

### Overview

| Accuracy | Training Time |
|---|---|
| ![Accuracy across all 22 runs](plots/accuracy.png) | ![Training time across all 22 runs](plots/training_time.png) |

---

### PS vs AllReduce — 2 workers, base serializer
_Runs 1, 2 (PS barrier) vs 11, 12 (AllReduce ring)_

| | dense_small | dense_large |
|---|---|---|
| Accuracy | ![](plots/cmp_ps_vs_ar_dense_small_accuracy.png) | ![](plots/cmp_ps_vs_ar_dense_large_accuracy.png) |
| Time | ![](plots/cmp_ps_vs_ar_dense_small_time.png) | ![](plots/cmp_ps_vs_ar_dense_large_time.png) |

---

### PS Worker Scaling — base serializer, barrier sync
_Runs 1–6: 2w/1s → 3w/1s → 3w/2s_

| | dense_small | dense_large |
|---|---|---|
| Accuracy | ![](plots/cmp_ps_worker_scaling_dense_small_accuracy.png) | ![](plots/cmp_ps_worker_scaling_dense_large_accuracy.png) |
| Time | ![](plots/cmp_ps_worker_scaling_dense_small_time.png) | ![](plots/cmp_ps_worker_scaling_dense_large_time.png) |

---

### AllReduce Worker Scaling — base serializer
_Runs 11–14: 2 workers vs 3 workers_

| | dense_small | dense_large |
|---|---|---|
| Accuracy | ![](plots/cmp_ar_worker_scaling_dense_small_accuracy.png) | ![](plots/cmp_ar_worker_scaling_dense_large_accuracy.png) |
| Time | ![](plots/cmp_ar_worker_scaling_dense_small_time.png) | ![](plots/cmp_ar_worker_scaling_dense_large_time.png) |

---

### Sparse vs Base Gradient — PS 2w/1s, barrier sync
_Runs 1, 2 (base) vs 7, 8 (sparse)_

| | dense_small | dense_large |
|---|---|---|
| Accuracy | ![](plots/cmp_sparse_vs_base_ps_dense_small_accuracy.png) | ![](plots/cmp_sparse_vs_base_ps_dense_large_accuracy.png) |
| Time | ![](plots/cmp_sparse_vs_base_ps_dense_small_time.png) | ![](plots/cmp_sparse_vs_base_ps_dense_large_time.png) |

---

### Sparse vs Base Gradient — AllReduce 2 workers
_Runs 11, 12 (base) vs 16, 17 (sparse)_

| | dense_small | dense_large |
|---|---|---|
| Accuracy | ![](plots/cmp_sparse_vs_base_ar_dense_small_accuracy.png) | ![](plots/cmp_sparse_vs_base_ar_dense_large_accuracy.png) |
| Time | ![](plots/cmp_sparse_vs_base_ar_dense_small_time.png) | ![](plots/cmp_sparse_vs_base_ar_dense_large_time.png) |

---

### AllReduce Sparse — Worker Scaling
_Runs 16, 17 (2w sparse) vs 19, 20 (3w sparse)_

| | dense_small | dense_large |
|---|---|---|
| Accuracy | ![](plots/cmp_ar_sparse_worker_scaling_dense_small_accuracy.png) | ![](plots/cmp_ar_sparse_worker_scaling_dense_large_accuracy.png) |
| Time | ![](plots/cmp_ar_sparse_worker_scaling_dense_small_time.png) | ![](plots/cmp_ar_sparse_worker_scaling_dense_large_time.png) |

---

### Barrier vs Non-blocking Sync — PS 2w/1s, base serializer
_Runs 1, 2 (barrier) vs 9, 10 (nonblocking wild)_

| | dense_small | dense_large |
|---|---|---|
| Accuracy | ![](plots/cmp_barrier_vs_nonblocking_dense_small_accuracy.png) | ![](plots/cmp_barrier_vs_nonblocking_dense_large_accuracy.png) |
| Time | ![](plots/cmp_barrier_vs_nonblocking_dense_small_time.png) | ![](plots/cmp_barrier_vs_nonblocking_dense_large_time.png) |

---

### Conv — AllReduce Worker Scaling
_Runs 22, 23: 2 workers vs 3 workers_

| Accuracy | Time |
|---|---|
| ![](plots/cmp_conv_ar_worker_scaling_conv_small_softmax_accuracy.png) | ![](plots/cmp_conv_ar_worker_scaling_conv_small_softmax_time.png) |

---

### Conv — PS vs AllReduce, 2 workers
_Runs 22 (AllReduce) vs 30 (PS barrier)_

| Accuracy | Time |
|---|---|
| ![](plots/cmp_conv_ps_vs_ar_conv_small_softmax_accuracy.png) | ![](plots/cmp_conv_ps_vs_ar_conv_small_softmax_time.png) |

---

### Conv — PS Worker Scaling
_Runs 30, 32: 2w/1s vs 3w/1s_

| Accuracy | Time |
|---|---|
| ![](plots/cmp_conv_ps_worker_scaling_conv_small_softmax_accuracy.png) | ![](plots/cmp_conv_ps_worker_scaling_conv_small_softmax_time.png) |

---

## What Gets Measured

| Field | Includes | Excludes |
|-------|----------|---------|
| `train_seconds` | `orchestrate()` → `session.wait()` | Docker build, dataset prep |
| `docker_start_seconds` | `docker compose up` + readiness wait | — |
| `eval_seconds` | PyTorch inference on test set | — |
| `accuracy` | Top-1 accuracy on 10k test set | — |

Pass/fail: `accuracy >= min_accuracy AND train_seconds <= max_train_seconds`.

## Models

| Name | Architecture |
|------|-------------|
| `dense_small` | 784 → 128 → 64 → 10 (Sigmoid) |
| `dense_large` | 784 → 256 → 128 → 64 → 10 (Sigmoid) |
| `conv_small_softmax` | Conv2d(16 filters, 4×4, stride 2) → flatten(2704) → 64 → 10 (Softmax), CE loss |

## Results Format

```json
{
  "run_num": 1,
  "model_name": "dense_small",
  "workers": 2, "servers": 1,
  "algorithm": "parameter_server",
  "serializer": "base", "sync": "barrier",
  "train_seconds": 24.2, "accuracy": 0.910,
  "passed": true, "error": null
}
```

Results (`.jsonl`) are gitignored. Plots (`plots/*.png`) are tracked.
