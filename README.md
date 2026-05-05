<p align="center">
  <img src="assets/logo.jpg" alt="O.N.O Logo" width="200"/>
</p>

<h1 align="center">Oxidized Neural Orchestra</h1>
<p align="center"><em>Distributed neural network training in Rust</em></p>

<p align="center">
  <img src="https://img.shields.io/badge/language-Rust-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/python-FFI-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/status-in%20development-yellow?style=flat-square" />
</p>

---

## Overview

**O.N.O** is a fully distributed system for training neural networks, built from scratch in Rust. It implements a Parameter Server architecture where workers train local models in parallel and synchronize gradients through centralized parameter servers.

The system exposes three interfaces:
- **`orchestui`** — an interactive TUI to configure and monitor training in real time
- **`orchestra-py`** — a Python-native module via PyO3 for programmatic use
- **`orchestrator`** — a headless Rust binary for Docker-based deployments

---

## Architecture

```
         Orchestrator
              |
       +------+------+
       |             |
       v             v
    Worker  <---->  Server
   (N nodes)       (M nodes)
```

- The **orchestrator** connects to all workers and servers, sends each node its configuration spec, and waits for training to complete.
- Each **worker** trains a local copy of the model on its data partition and synchronizes gradients with the parameter servers.
- Each **parameter server** holds a partition of the model parameters and applies gradient updates.

Both workers and servers run the same `node` binary. The orchestrator sends a spec at connection time that determines which role the node takes for that session.

---

## Running Locally

### Prerequisites
- Rust toolchain (`rustup`)
- Python 3.12+
- `maturin` (`pipx install maturin`)
- Docker + Docker Compose (for distributed mode)

---

### Option 1 — TUI (`orchestui`)

The interactive terminal dashboard. Requires `model.json` and `training.json` config files.
```bash
cargo run -p orchestui
```

When the TUI opens, enter the paths to your config files (leave blank to use `model.json` / `training.json` in the current directory):
```
model.json path:    orchestui/model.json
training.json path: orchestui/training.json
```

Press `?` at either prompt to see an inline config example. Example config files are provided in `orchestui/`.

---

### Option 2 — Python (`orchestra-py`)

Install the module and run the example script — containers are started first via Docker:
```bash
cd orchestra-py
python3 -m venv ../.venv
source ../.venv/bin/activate
maturin develop
cd ..
python3 docker/compose_up.py --workers 3 --servers 3
source .venv/bin/activate && python3 orchestra-py/local.py
```

---

### Option 3 — Headless Rust (`orchestrator`)

Start the node containers, then run the orchestrator binary against your config files:
```bash
python3 docker/compose_up.py --workers N --servers M --release
cargo run -p orchestrator -- model.json training.json
```

---

## Docker Configuration

All node containers (workers and servers) are started with:
```bash
python3 docker/compose_up.py --workers N --servers M [--release]
```

This script:
1. Generates `compose.yaml` via `docker/gen_compose.py`
2. Fills `/etc/hosts` with `worker-*` / `server-*` entries (requires sudo once)
3. Runs `docker compose up --build -d --remove-orphans`

All containers use the same `node/Dockerfile`. Server addresses start at port `40000`, worker addresses at `50000`. Address lists are generated automatically — no hardcoding needed.

---

## Config Files

### `model.json`
```json
{
  "layers": [
    { "dense": { "output_size": 8, "init": "kaiming", "act_fn": { "sigmoid": { "amp": 1.0 } } } },
    { "dense": { "output_size": 4, "init": "kaiming", "act_fn": { "sigmoid": { "amp": 1.0 } } } },
    { "dense": { "output_size": 1, "init": "kaiming" } }
  ]
}
```

### `training.json`
```json
{
  "worker_addrs": ["127.0.0.1:50000", "127.0.0.1:50001", "127.0.0.1:50002"],
  "algorithm": {
    "parameter_server": {
      "server_addrs": ["127.0.0.1:40000", "127.0.0.1:40001"],
      "synchronizer": "barrier",
      "store": "blocking"
    }
  },
  "dataset": {
    "src": {
      "inline": {
        "samples": [1.0, 2.0, 3.0, 4.0],
        "labels": [2.0, 4.0, 6.0, 8.0]
      }
    },
    "x_size": 1,
    "y_size": 1
  },
  "optimizer": { "gradient_descent": { "lr": 0.01 } },
  "loss_fn": "mse",
  "batch_size": 4,
  "max_epochs": 500,
  "offline_epochs": 0,
  "seed": 42,
  "early_stopping": { "tolerance": 1e-4 }
}
```

Synchronizer options: `"barrier"` | `"non_blocking"`  
Store options: `"blocking"` | `"wild"`  
`seed`, `serializer`, `early_stopping`, and `act_fn` are optional — omit them to use defaults.  
For a local dataset use `"src": { "local": { "samples_path": "...", "labels_path": "..." } }` instead of `inline`.

---

## Python API
```python
from orchestra import Sequential, orchestrate, parameter_server
from orchestra.arch import Dense
from orchestra.activations import Sigmoid
from orchestra.initialization import Kaiming
from orchestra.datasets import InlineDataset
from orchestra.optimizers import GradientDescent
from orchestra.loss_fns import Mse
from orchestra.sync import BarrierSync
from orchestra.store import BlockingStore

model = Sequential([
    Dense(8, Kaiming(), Sigmoid()),
    Dense(4, Kaiming(), Sigmoid()),
    Dense(1, Kaiming()),
])

samples = [1.0, 2.0, 3.0, 4.0]
labels  = [2.0, 4.0, 6.0, 8.0]

training = parameter_server(
    worker_addrs=["127.0.0.1:50000", "127.0.0.1:50001", "127.0.0.1:50002"],
    server_addrs=["127.0.0.1:40000", "127.0.0.1:40001"],
    dataset=InlineDataset(samples, labels, x_size=1, y_size=1),
    optimizer=GradientDescent(lr=0.01),
    loss_fn=Mse(),
    sync=BarrierSync(),
    store=BlockingStore(),
    max_epochs=500,
    batch_size=4,
)

session = orchestrate(model, training)
trained = session.wait()
trained.save_safetensors("weights.safetensors")
```

---

## What's Implemented

| Feature | Status |
|---|---|
| Neural network library (dense layers, activations, optimizers) | ✅ |
| Conv2d convolutional layers | ✅ |
| Parameter Server distributed training | ✅ |
| Barrier and non-blocking synchronization | ✅ |
| Early stopping | ✅ |
| Sparse gradient compression | ✅ |
| Manual session stop (`session.stop()`) | ✅ |
| TUI dashboard (`orchestui`) | ✅ |
| Python FFI via PyO3 (`orchestra-py`) | ✅ |
| Docker deployment | ✅ |
| All-Reduce strategy (training runs; final model retrieval not implemented) | 🔲 |

---

## Team

| Name | Student ID |
|---|---|
| Lorenzo Minervino | 107863 |
| Marcos Bianchi | 108921 |
| Alejo Ordoñez | 108397 |

**Universidad de Buenos Aires**  
Faculty of Engineering — Computer Engineering

---

## License

All rights reserved to the authors until project completion.