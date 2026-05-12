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
# All 22 combinations (~30–35 min)
.venv/bin/python benchmarks/mnist_e2e.py

# Specific numbered runs
.venv/bin/python benchmarks/mnist_e2e.py --runs 1 11 22

# Keep containers after run / force image rebuild
.venv/bin/python benchmarks/mnist_e2e.py --keep-containers --rebuild
```

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
| 23 | conv_small_softmax | AllReduce | 3 | 0 | base | — | 150 |
| 30 | conv_small_softmax | PS | 2 | 1 | base | barrier | 100 |
| 32 | conv_small_softmax | PS | 3 | 1 | base | barrier | 250 |

**Not included:** `conv_large_softmax` (Conv2d 32 filters) and conv sparse runs.
`conv_large` requires ~15s/epoch making a 200-epoch run ~50 min; the sparse serializer
adds a 6–7× per-epoch overhead for conv tensors. Both dimensions are covered by dense
runs (large models) and dense sparse runs (sparse gradient compression) respectively.
`conv_small_softmax` is sufficient to validate conv support end-to-end.

## Comparison Plots

Each comparison generates one chart per model per metric (`{name}_{model}_accuracy.png` / `{name}_{model}_time.png`).

| Group | Title | Runs |
|-------|-------|------|
| `cmp_ps_vs_ar` | PS vs AllReduce — 2w, base | 1, 2, 11, 12 |
| `cmp_ps_worker_scaling` | PS worker scaling — base, barrier | 1–6 |
| `cmp_ar_worker_scaling` | AllReduce worker scaling — base | 11–14 |
| `cmp_sparse_vs_base_ps` | Sparse vs base — PS 2w/1s, barrier | 1, 2, 7, 8 |
| `cmp_sparse_vs_base_ar` | Sparse vs base — AllReduce 2w | 11, 12, 16, 17 |
| `cmp_ar_sparse_worker_scaling` | AllReduce sparse — worker scaling | 16, 17, 19, 20 |
| `cmp_barrier_vs_nonblocking` | Barrier vs non-blocking — PS 2w/1s, base | 1, 2, 9, 10 |
| `cmp_conv_ar_worker_scaling` | Conv AllReduce worker scaling — base | 22, 23 |
| `cmp_conv_ps_vs_ar` | Conv PS vs AllReduce — 2w base | 22, 30 |
| `cmp_conv_ps_worker_scaling` | Conv PS worker scaling — base, barrier | 30, 32 |

## Latest Results

### Accuracy
![Accuracy](plots/accuracy.png)

### Training Time
![Training Time](plots/training_time.png)

### PS vs AllReduce
| | dense_small | dense_large |
|---|---|---|
| Accuracy | ![](plots/cmp_ps_vs_ar_dense_small_accuracy.png) | ![](plots/cmp_ps_vs_ar_dense_large_accuracy.png) |
| Time | ![](plots/cmp_ps_vs_ar_dense_small_time.png) | ![](plots/cmp_ps_vs_ar_dense_large_time.png) |

### PS Worker Scaling
| | dense_small | dense_large |
|---|---|---|
| Accuracy | ![](plots/cmp_ps_worker_scaling_dense_small_accuracy.png) | ![](plots/cmp_ps_worker_scaling_dense_large_accuracy.png) |
| Time | ![](plots/cmp_ps_worker_scaling_dense_small_time.png) | ![](plots/cmp_ps_worker_scaling_dense_large_time.png) |

### AllReduce Worker Scaling
| | dense_small | dense_large |
|---|---|---|
| Accuracy | ![](plots/cmp_ar_worker_scaling_dense_small_accuracy.png) | ![](plots/cmp_ar_worker_scaling_dense_large_accuracy.png) |
| Time | ![](plots/cmp_ar_worker_scaling_dense_small_time.png) | ![](plots/cmp_ar_worker_scaling_dense_large_time.png) |

### Sparse vs Base — PS
| | dense_small | dense_large |
|---|---|---|
| Accuracy | ![](plots/cmp_sparse_vs_base_ps_dense_small_accuracy.png) | ![](plots/cmp_sparse_vs_base_ps_dense_large_accuracy.png) |
| Time | ![](plots/cmp_sparse_vs_base_ps_dense_small_time.png) | ![](plots/cmp_sparse_vs_base_ps_dense_large_time.png) |

### Sparse vs Base — AllReduce
| | dense_small | dense_large |
|---|---|---|
| Accuracy | ![](plots/cmp_sparse_vs_base_ar_dense_small_accuracy.png) | ![](plots/cmp_sparse_vs_base_ar_dense_large_accuracy.png) |
| Time | ![](plots/cmp_sparse_vs_base_ar_dense_small_time.png) | ![](plots/cmp_sparse_vs_base_ar_dense_large_time.png) |

### AllReduce Sparse — Worker Scaling
| | dense_small | dense_large |
|---|---|---|
| Accuracy | ![](plots/cmp_ar_sparse_worker_scaling_dense_small_accuracy.png) | ![](plots/cmp_ar_sparse_worker_scaling_dense_large_accuracy.png) |
| Time | ![](plots/cmp_ar_sparse_worker_scaling_dense_small_time.png) | ![](plots/cmp_ar_sparse_worker_scaling_dense_large_time.png) |

### Barrier vs Non-blocking
| | dense_small | dense_large |
|---|---|---|
| Accuracy | ![](plots/cmp_barrier_vs_nonblocking_dense_small_accuracy.png) | ![](plots/cmp_barrier_vs_nonblocking_dense_large_accuracy.png) |
| Time | ![](plots/cmp_barrier_vs_nonblocking_dense_small_time.png) | ![](plots/cmp_barrier_vs_nonblocking_dense_large_time.png) |

### Conv AllReduce Worker Scaling
| | conv_small_softmax |
|---|---|
| Accuracy | ![](plots/cmp_conv_ar_worker_scaling_conv_small_softmax_accuracy.png) |
| Time | ![](plots/cmp_conv_ar_worker_scaling_conv_small_softmax_time.png) |

### Conv PS vs AllReduce
| | conv_small_softmax |
|---|---|
| Accuracy | ![](plots/cmp_conv_ps_vs_ar_conv_small_softmax_accuracy.png) |
| Time | ![](plots/cmp_conv_ps_vs_ar_conv_small_softmax_time.png) |

### Conv PS Worker Scaling
| | conv_small_softmax |
|---|---|
| Accuracy | ![](plots/cmp_conv_ps_worker_scaling_conv_small_softmax_accuracy.png) |
| Time | ![](plots/cmp_conv_ps_worker_scaling_conv_small_softmax_time.png) |

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
