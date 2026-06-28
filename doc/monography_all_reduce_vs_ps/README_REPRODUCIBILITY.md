# Reproducibility — Parameter Server vs All-Reduce in O.N.O.

This folder is **self-contained**. It drives the O.N.O. distributed trainer through
its native Python binding (`orchestra`, built from the repo via PyO3) and a Docker
cluster, runs the monography's experiment matrix, and produces every figure, table
and the final PDF. It does **not** depend on the repo's `benchmarks/` harness.

## 0. Requirements (already present on the reference machine)

- Rust toolchain (the node image compiles `node/Dockerfile`).
- Docker + Docker Compose v2, with the user in the `docker` group (no sudo).
- `/etc/hosts` must map `node-0 … node-N` to `127.0.0.1` (set once via the repo's
  `docker/fill_hosts.py`; the cluster reaches containers through published ports).
- Python venv at repo root `../../.venv` with `orchestra` installed (`maturin develop`
  inside `orchestra-py/` if the binding is stale), plus `numpy`, `torch`, `matplotlib`,
  `safetensors`.
- TeX Live with `latexmk`, `IEEEtran`, `pgfplots`, `biblatex`+`biber` (or BibTeX),
  `booktabs`, `siunitx`, `cleveref`, `microtype`.

## 1. Layout

```
scripts/      run_experiment.py · ono_harness.py · download_datasets.py
              aggregate.py · make_figures.py · make_tables.py
configs/      experiment matrices (JSON) — exp_a_sanity.json, exp_b_*.json, ...
datasets/     MNIST (.bin float32) · EMNIST (optional)
results/raw/  one JSONL line per run (full per-epoch loss history)
results/processed/  results.csv + summary_*.csv (the analysed tables)
figures/      *.pdf figures embedded in the paper
tables/       *.tex tables \input by main.tex
logs/         per-campaign human-readable logs
main.tex · references.bib · Makefile · main.pdf
```

## 2. Datasets

```
make datasets            # MNIST only
make datasets EMNIST=1   # also fetch + convert EMNIST
```

| Dataset | Source | Format on disk | Shape | Classes | Train/Test | Norm |
|---|---|---|---|---|---|---|
| MNIST | CVDF mirror of LeCun's IDX | raw float32 `.bin`: samples 784-flatten, labels one-hot 10 | 28×28×1 | 10 | 60000 / 10000 | /255 → [0,1] |
| EMNIST (optional) | NIST EMNIST | same `.bin` conversion | 28×28×1 | varies by split | per split | /255 → [0,1] |

`batch_size` in O.N.O. is **per worker**; the effective global batch is
`workers × batch_size`. This is recorded explicitly in `results.csv`.

## 3. Run the campaign

```
# Smoke (≈ tiny, validates the pipeline end-to-end):
scripts/run_experiment.py --smoke

# A full experiment matrix:
scripts/run_experiment.py --config configs/exp_a_sanity.json
scripts/run_experiment.py --config configs/exp_b_fixed_global.json
scripts/run_experiment.py --config configs/exp_b_fixed_perworker.json
```

Each run appends one line to `results/raw/<name>.jsonl`. Runs never abort the
campaign on a single failure — a failed run is recorded with `status="error"`
and the traceback, and the matrix continues.

## 4. Analyse, plot, compile

```
scripts/aggregate.py      # results/raw/*.jsonl -> results/processed/results.csv + summaries
make plots                # figures/*.pdf
make tables               # tables/*.tex
make pdf                  # main.pdf
```

## 5. Experiment matrix (scope)

- **Exp A — sanity (MNIST, nielsen):** PyTorch baseline, All-Reduce, Parameter
  Server (barrier + blocking), Strategy Switch.
- **Exp B — AR vs PS (MNIST, nielsen, 3 and 5 nodes):** *fixed global batch* (256)
  for a fair convergence comparison, plus *fixed per-worker batch* for raw throughput.
- **Optional (if time):** scalable MLPs (`mlp_small/medium/large`) for communication
  pressure, and EMNIST.

Excluded by design on this machine: the `non_blocking` PS synchronizer (unstable
under load), 7-node runs (an 8-core host saturates), Fashion-MNIST and the straggler
study. The cluster is **simulated on a single 8-core machine** via Docker, so scaling
numbers measure *logical* scaling under shared physical resources, not multi-machine
scaling — see the paper's Limitations.

## 6. Honesty notes

- `test_loss` is **not** computed by the engine (only per-epoch train loss + final
  test accuracy); that CSV column is intentionally `NA`.
- All claims in the paper are scoped to "the environment evaluated".
