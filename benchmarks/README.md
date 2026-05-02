# MNIST End-to-End Benchmark

Runs distributed MNIST training through the full O.N.O stack (Docker → workers → parameter server → orchestrator), measures performance metrics, and generates reproducible comparison plots.

## Setup

All dependencies are in the repo-root `.venv` (torch, safetensors, orchestra, matplotlib). No extra install needed.

```bash
# Verify the venv works
.venv/bin/python -c "import orchestra, torch, matplotlib; print('OK')"
```

If you don't have the venv yet:

```bash
cd orchestra-py && maturin develop --release && cd ..
pip install torch safetensors matplotlib  # or use the existing .venv
```

## Running

```bash
# Quick smoke test — validates the full pipeline (~2 min)
.venv/bin/python benchmarks/mnist_e2e.py --profile smoke

# Full benchmark — compares configs, timing, and accuracy (~10-20 min)
.venv/bin/python benchmarks/mnist_e2e.py --profile benchmark
```

### Options

| Flag | Effect |
|---|---|
| `--profile smoke\|benchmark` | Which config profile to run (default: smoke) |
| `--output path/to/file.jsonl` | Custom results file path |
| `--keep-containers` | Don't stop Docker containers after the run |
| `--rebuild` | Force Docker image rebuild (`--no-cache`) |
| `--config path/to/config.json` | Custom config file (default: `mnist_configs.json`) |

## Latest Results

> These images are regenerated automatically on every run. Open this file in VS Code or any local Markdown viewer to see fresh results.

### Smoke

![Smoke Results](results/latest/smoke_results.png)

### Benchmark

![Benchmark Results](results/latest/benchmark_results.png)

---

## What Gets Measured

| Field | What's included | What's excluded |
|---|---|---|
| `train_seconds` | `orchestrate()` call → `session.wait()` returns | Docker build, container startup, dataset prep |
| `docker_start_seconds` | `docker compose up` + 3s readiness wait | — |
| `eval_seconds` | PyTorch inference on test set | — |
| `accuracy` | Top-1 accuracy on N test samples | — |

**Pass/fail per run**: `accuracy >= min_accuracy AND train_seconds <= max_train_seconds`.

## Results Format

Each run appends one line to a `.jsonl` file in `results/` (gitignored):

```json
{
  "run_id": "2026-05-01T22-25-34_dense_tiny_ps_1w_1s_base_barrier",
  "profile": "smoke",
  "model_name": "dense_tiny",
  "training_name": "ps_1w_1s_base_barrier",
  "workers": 1,
  "servers": 1,
  "serializer": "base",
  "sync": "barrier",
  "store": "blocking",
  "max_epochs": 5,
  "batch_size": 256,
  "lr": 0.5,
  "train_samples": 1000,
  "test_samples": 500,
  "docker_start_seconds": 21.6,
  "train_seconds": 0.06,
  "eval_seconds": 1.2,
  "accuracy": 0.204,
  "min_accuracy": 0.15,
  "max_train_seconds": 180,
  "passed": true,
  "error": null
}
```

## Profiles

| Profile | Train | Test | Purpose | Expected time |
|---|---|---|---|---|
| `smoke` | full (60 k) | full (10 k) | Pipeline validation — 15 epochs, min 75% accuracy | ~5–10 min |
| `benchmark` | full (60 k) | full (10 k) | Scaling + convergence comparison, 30–100 epochs | ~30–90 min |

## Models

Dense-only. Conv2d configs are excluded until that implementation is stable.

| Name | Architecture | Used in |
|---|---|---|
| `dense_tiny` | 784 → 64 → 10 (Sigmoid) | smoke + benchmark |
| `dense_small` | 784 → 128 → 64 → 10 (Sigmoid) | benchmark |

## Training Configs

| Name | Workers | Servers | Serializer | Sync | Early stop |
|---|---|---|---|---|---|
| `ps_1w_1s_base_barrier` | 1 | 1 | Base | Barrier | — |
| `ps_2w_1s_base_barrier` | 2 | 1 | Base | Barrier | — |
| `ps_3w_1s_base_barrier` | 3 | 1 | Base | Barrier | — |
| `ps_3w_2s_base_barrier` | 3 | 2 | Base | Barrier | — |
| `ps_2w_1s_base_barrier` + `early_stopping_tolerance` | 2 | 1 | Base | Barrier | 0.001 |
| `ps_3w_2s_base_barrier` + `early_stopping_tolerance` | 3 | 2 | Base | Barrier | 0.001 |

The `3w_1s` vs `3w_2s` comparison reveals whether a single parameter server becomes a bottleneck with 3 workers. The early-stopping variants (benchmark only) run up to 100 epochs but stop automatically when epoch-to-epoch MSE improvement drops below the tolerance.

## Customizing

Edit `mnist_configs.json` to add runs, change epochs, thresholds, or lr. The structure:

```json
{
  "profiles": {
    "smoke": {
      "train_samples": 1000,
      "test_samples": 500,
      "runs": [
        {
          "name": "my_run",
          "model": "dense_tiny",
          "training": "ps_1w_1s_base_barrier",
          "max_epochs": 5,
          "batch_size": 256,
          "lr": 0.5,
          "offline_epochs": 0,
          "min_accuracy": 0.15,
          "max_train_seconds": 180
        }
      ]
    }
  }
}
```

## Notes

- `early_stopping_tolerance` is always `None` — per-worker early stop deadlocks with BarrierSync
- `/etc/hosts` needs `worker-*`/`server-*` entries; managed automatically via `docker/fill_hosts.py` (requires sudo once if missing)
- Subset dataset files are cached in `results/data/` after first run — not recreated on subsequent runs
- Results files (`*.jsonl`) and plots (`results/latest/`) are gitignored; only the script and config are tracked
