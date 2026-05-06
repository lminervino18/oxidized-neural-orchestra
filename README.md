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

**O.N.O** is a fully distributed system for training neural networks, built from scratch in Rust. It supports two distributed training algorithms:

- **Parameter Server** — workers push gradients to centralized parameter servers, which apply updates and return the new parameters.
- **All-Reduce** — workers exchange and reduce gradients directly with each other, with no central server.

The system exposes three interfaces:
- **`orchestui`** — an interactive TUI to configure and monitor training in real time
- **`orchestra-py`** — a Python-native module via PyO3 for programmatic use
- **`orchestrator`** — a headless Rust binary for Docker-based deployments

---

## Architecture

The **orchestrator** connects to all nodes, sends each one its configuration spec, and waits for training to complete. All nodes run the same `node` binary — the spec received at connection time determines whether a node acts as a worker or a parameter server.

### Parameter Server

```
         Orchestrator
              |
       +------+------+
       |             |
       v             v
    Worker  <---->  Server
   (N nodes)       (M nodes)
```

Workers train locally on their data partition and push gradients to the parameter servers. Servers apply updates and return the new parameters. Model parameters are sharded across servers.

### All-Reduce

```
        Orchestrator
             |
    +--------+--------+
    |        |        |
    v        v        v
 Worker  Worker  Worker
    |        |        |
    +--------+--------+
         (ring reduce)
```

Workers exchange and reduce gradients directly with each other — no parameter server needed. Each worker ends up with the same averaged gradient and applies it locally.

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

Two algorithm options — pick one:

**Parameter Server:**
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

**All-Reduce:**
```json
{
  "worker_addrs": ["127.0.0.1:50000", "127.0.0.1:50001", "127.0.0.1:50002"],
  "algorithm": "all_reduce",
  "dataset": { ... },
  "optimizer": { "gradient_descent": { "lr": 0.01 } },
  "loss_fn": "mse",
  "batch_size": 4,
  "max_epochs": 500,
  "offline_epochs": 0
}
```

Synchronizer options (PS only): `"barrier"` | `"non_blocking"`  
Store options (PS only): `"blocking"` | `"wild"`  
`seed`, `serializer`, `early_stopping`, and `act_fn` are optional — omit them to use defaults.  
For a local dataset use `"src": { "local": { "samples_path": "...", "labels_path": "..." } }` instead of `inline`.

---

## Python API

### Parameter Server
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

### All-Reduce
```python
from orchestra import Sequential, orchestrate, all_reduce
from orchestra.arch import Dense
from orchestra.initialization import Kaiming
from orchestra.datasets import InlineDataset
from orchestra.optimizers import GradientDescent
from orchestra.loss_fns import Mse

model = Sequential([Dense(8, Kaiming()), Dense(1, Kaiming())])

training = all_reduce(
    worker_addrs=["127.0.0.1:50000", "127.0.0.1:50001", "127.0.0.1:50002"],
    dataset=InlineDataset(samples, labels, x_size=1, y_size=1),
    optimizer=GradientDescent(lr=0.01),
    loss_fn=Mse(),
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
| Global architecture | ✅ |
| Parameter Server | ✅ |
| All-Reduce | ✅ |
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