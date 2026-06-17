# Strategy Benchmarks

Compares the three distributed strategies — **parameter server**, **all-reduce** and **strategy switch** — on two models (**LeNet5** and **Nielsen MNIST**) across four focused suites. Each suite states what it measures and what it does not.

## Models

- **Nielsen MNIST**: `28×28×1 → conv(20, 5×5) → maxpool(2×2) → dense(100) → dense(10) → softmax`.
- **LeNet5**: `conv(6, 5×5, pad2) → maxpool → conv(16, 5×5) → maxpool → dense(120) → dense(84) → dense(10)`, tanh + softmax.

## Running

```bash
.venv/bin/python benchmarks/run_issue_benchmarks.py                 # all suites, both models
.venv/bin/python benchmarks/run_issue_benchmarks.py --suite convergence
.venv/bin/python benchmarks/run_issue_benchmarks.py --suite scalability --model lenet5
.venv/bin/python benchmarks/run_issue_benchmarks.py --plots-only    # rebuild plots/README from history
```

Partial runs only re-run and re-plot the selected suite/model; every other suite keeps its previous results and figures.

All-reduce worker scale: [3, 5, 7] (configurable in `issue/suites.py`; the issue suggests 3/7/11 — kept lighter to fit one host).

_Last full run: 4h 17m 40s (2026-06-16 21:05)._

## Convergence

**Measures:** loss vs epoch and final test accuracy per strategy/topology.
**Does NOT measure:** wall-clock speed.

| Model | Strategy | Topology | Epochs | Final loss | Accuracy |
|---|---|---|---|---|---|
| lenet5 | AR | 3w | 60 | 0.00517 | 0.986 |
| lenet5 | AR | 5w | 60 | 0.2 | 0.486 |
| lenet5 | AR | 7w | 21 | 1.44 | 0.101 |
| lenet5 | PS | 3w/2s | 60 | 0.00435 | 0.115 |
| lenet5 | SS | 3w/2s | 60 | 0.2 | 0.486 |
| nielsen | AR | 3w | 60 | 0.0101 | 0.966 |
| nielsen | AR | 5w | 60 | 1.41 | 0.113 |
| nielsen | AR | 7w | 23 | 1.44 | 0.103 |
| nielsen | PS | 3w/2s | 60 | 0.00971 | 0.103 |
| nielsen | SS | 3w/2s | — | — | — |

![](plots/convergence_loss_nielsen.png)
![](plots/convergence_loss_lenet5.png)
![](plots/convergence_accuracy_nielsen.png)
![](plots/convergence_accuracy_lenet5.png)

## Execution speed

**Measures:** epochs/sec on a small subset (no convergence). Compares raising `offline_epochs` vs raising `batch_size`.
**Does NOT measure:** accuracy or convergence.

| Model | Strategy | Topology | offline | batch | Epochs/sec |
|---|---|---|---|---|---|
| lenet5 | AR | 3w | 0 | 64 | 0.73 |
| lenet5 | AR | 3w | 4 | 64 | 0.725 |
| lenet5 | AR | 3w | 0 | 256 | 0.709 |
| nielsen | AR | 3w | 0 | 64 | 0.985 |
| nielsen | AR | 3w | 4 | 64 | 0.989 |
| nielsen | AR | 3w | 0 | 256 | 0.925 |

![](plots/execution_speed_nielsen.png)
![](plots/execution_speed_lenet5.png)

## Convergence speed

**Measures:** loss reduction/sec and accuracy/sec under one shared budget (same epochs and params; only the strategy changes).
**Does NOT measure:** peak accuracy.

| Model | Strategy | Topology | Loss/sec | Accuracy/sec |
|---|---|---|---|---|
| lenet5 | AR | 3w | 0.000261 | 0.00103 |
| lenet5 | PS | 3w/2s | 0.000218 | 0.000127 |
| lenet5 | SS | 3w/2s | 0.000179 | 0.000363 |
| nielsen | AR | 3w | 0.000315 | 0.00133 |
| nielsen | PS | 3w/2s | 0.000317 | 0.000161 |
| nielsen | SS | 3w/2s | — | — |

![](plots/convergence_speed_nielsen.png)
![](plots/convergence_speed_lenet5.png)

## Scalability

**Measures:** how throughput (epochs/sec) changes as nodes increase.
**Does NOT measure:** convergence (re-uses the speed budget).

| Model | Strategy | Workers | Epochs/sec |
|---|---|---|---|
| lenet5 | AR | 3 | 0.712 |
| lenet5 | AR | 5 | 0.902 |
| lenet5 | AR | 7 | 1.07 |
| lenet5 | PS | 2 | 0.533 |
| lenet5 | PS | 3 | 0.748 |
| nielsen | AR | 3 | 0.928 |
| nielsen | AR | 5 | 1.18 |
| nielsen | AR | 7 | 1.33 |
| nielsen | PS | 2 | 0.726 |
| nielsen | PS | 3 | 0.979 |

![](plots/scalability_nielsen.png)
![](plots/scalability_lenet5.png)

## Raw results

- Per-run records: `results/*.jsonl` (append-only, gitignored).
- Flattened table: `results/summary.csv`.
- Trained weights: `results/artifacts/*.safetensors`.

**Metrics:** `epochs_per_sec = epochs / train_seconds`; `loss_per_sec = (first_loss − final_loss) / train_seconds`; `accuracy_per_sec = accuracy / train_seconds`.
