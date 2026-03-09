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
./orchestui/run.sh
```

When the TUI opens, enter the paths to your config files:
```
model.json path:    orchestui/model.json
training.json path: orchestui/training.json
```

Example config files are provided in `orchestui/`.

---

### Option 2 — Python (`orchestra-py`)

Install the module and run locally — workers and servers are spawned automatically:
```bash
cd orchestra-py
python3 -m venv ../.venv
source ../.venv/bin/activate
maturin develop
cd ..
source .venv/bin/activate && python3 orchestra-py/local.py
```

Or via Docker (workers and servers must be running first):
```bash
./entities_up.sh
./orchestrator_up.sh orchestra-py
```

---

### Option 3 — Headless Rust (`orchestrator`)
```bash
./entities_up.sh
./orchestrator_up.sh orchestrator
```

---

## Docker Configuration

All Docker deployments read from `docker/config.json`:
```json
{
  "release": false,
  "servers": 2,
  "workers": 3
}
```

- `release` — compile in release mode
- `servers` — number of parameter server containers
- `workers` — number of worker containers

Worker and server addresses are generated automatically from this config — no hardcoding needed.

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
    "src": { "inline": { "data": [1.0,2.0, 2.0,4.0, 3.0,6.0, 4.0,8.0] } },
    "x_size": 1,
    "y_size": 1
  },
  "optimizer": { "gradient_descent": { "lr": 0.01 } },
  "loss_fn": "mse",
  "batch_size": 4,
  "max_epochs": 500,
  "offline_epochs": 0
}
```

Synchronizer options: `"barrier"` | `"non_blocking"`  
Store options: `"blocking"` | `"wild"`  
`seed` and `act_fn` are optional — omit them to use defaults.

---

## Python API
```python
import orchestra
from orchestra import Sequential, orchestrate
from orchestra.arch import Dense
from orchestra.activations import Sigmoid
from orchestra.initialization import Kaiming
from orchestra.datasets import InlineDataset
from orchestra.optimizers import GradientDescent
from orchestra.sync import BarrierSync
from orchestra.store import BlockingStore

model = Sequential([
    Dense(8, Kaiming(), Sigmoid()),
    Dense(4, Kaiming(), Sigmoid()),
    Dense(1, Kaiming()),
])

training = orchestra.parameter_server(
    worker_addrs=["127.0.0.1:50000", "127.0.0.1:50001", "127.0.0.1:50002"],
    server_addrs=["127.0.0.1:40000", "127.0.0.1:40001"],
    dataset=InlineDataset(data, x_size=1, y_size=1),
    optimizer=GradientDescent(lr=0.01),
    sync=BarrierSync(),
    store=BlockingStore(),
    max_epochs=500,
    batch_size=4,
)

session = orchestrate(model, training)
trained = session.wait()
trained.save("weights.csv", output_sizes=[8, 4, 1], input_size=1)
```

---

## What's Implemented

| Feature | Status |
|---|---|
| Neural network library (dense layers, activations, optimizers) | ✅ |
| Parameter Server distributed training | ✅ |
| Barrier and non-blocking synchronization | ✅ |
| TUI dashboard (`orchestui`) | ✅ |
| Python FFI via PyO3 (`orchestra-py`) | ✅ |
| Docker deployment | ✅ |
| All-Reduce strategy | 🔲 |
| Strategy Switch | 🔲 |

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