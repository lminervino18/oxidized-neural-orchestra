# orchestra-py

Python bindings to the **Oxidized Neural Orchestra (ONO)** distributed-training engine.
Built with [maturin](https://www.maturin.rs/) and [PyO3](https://pyo3.rs/), the extension
compiles the Rust `orchestrator` crate into a native module and exposes a high-level Python
API to define models, configure distributed training, launch runs, and retrieve trained
parameters.

The Python package is named **`orchestra`**; the compiled extension lives at
`orchestra._orchestra` and is re-exported through thin Python wrappers
(`orchestra.arch`, `orchestra.activations`, `orchestra.optimizers`, …).

## Installation

Build and install the extension from `orchestra-py/` (or the workspace root):

```bash
# Development build (fast to iterate, installs into the active venv)
maturin develop --release

# Or an editable install
pip install -e orchestra-py/
```

`maturin develop` compiles `src/*.rs` into `orchestra/_orchestra*.so` and makes the
`orchestra` package importable.

## How it works

Training runs across a set of **`node`** processes (local or remote) listening on the
addresses you pass in. The Python process is the **orchestrator**: it connects to those
nodes, ships the model and config, and streams progress back.

**Node roles are assigned at runtime** — you do not pre-label a node as "worker" or
"server". You provide a single `addrs` list; the algorithm decides:

- `all_reduce` — every node is a worker; gradients are averaged collectively.
- `parameter_server` — `nservers` of the nodes act as parameter servers, the rest as
  workers (`nservers` must be **less than** `len(addrs)`).
- `strategy_switch` — starts as all-reduce, then promotes `nservers` nodes to parameter
  servers once loss improvement stalls.

A run is three steps:

1. **Define the model** — a `Sequential` stack of layers.
2. **Configure training** — pick a strategy, dataset, optimizer, loss, and hyperparameters.
3. **Orchestrate** — `orchestrate(model, training)` connects to the nodes and returns a
   `Session`; `session.wait()` blocks until training finishes and returns the trained model.

---

## Quick start

XOR trained with all-reduce across two nodes:

```python
from orchestra import Sequential, orchestrate, all_reduce
from orchestra.arch import Dense
from orchestra.activations import Sigmoid
from orchestra.initialization import Kaiming
from orchestra.datasets import InlineDataset
from orchestra.optimizers import GradientDescent
from orchestra.loss_fns import Mse

dataset = InlineDataset(
    samples=[0., 0., 0., 1., 1., 0., 1., 1.],
    labels=[0., 1., 1., 0.],
    x_size=2,
    y_size=1,
)

model = Sequential([
    Dense(4, Kaiming(), act_fn=Sigmoid()),
    Dense(1, Kaiming()),
])

training = all_reduce(
    addrs=["node-0:40000", "node-1:40001"],
    dataset=dataset,
    optimizer=GradientDescent(lr=1.0),
    loss_fn=Mse(),
    max_epochs=500,
    batch_size=4,
    seed=42,
)

session = orchestrate(model, training)
trained = session.wait()

trained.save_safetensors("xor.safetensors")
```

> **Config schema.** These Python builders map 1:1 onto the JSON `model.json` /
> `training.json` config. For the underlying schema — every allowed value and its
> constraints — see **[`../docs/config-schema.md`](../docs/config-schema.md)**.
> The tables below give the Python spelling of the same options.

---

## Model definition

### `Sequential(layers)`

An ordered stack of layers. Requires at least one layer. When a `Conv2d`/`MaxPooling`
stage is followed by a `Dense` layer, its output is flattened automatically.

| Layer | Signature |
|-------|-----------|
| `Dense` | `Dense(output_size, init, act_fn=None)` |
| `Conv2d` | `Conv2d(input_dim, kernel_dim, stride, padding, init, act_fn=None)` |
| `MaxPooling` | `MaxPooling(input_dim, filter_size, stride, padding, act_fn=None)` |

- `output_size`: number of output neurons (`> 0`).
- `input_dim`: `(in_channels, height, width)`, all `> 0`.
- `kernel_dim`: `(filters, in_channels, kernel_size)`, all `> 0` (square kernel).
- `filter_size` / `stride`: `> 0`. `padding`: zero-padding per spatial side (`>= 0`).
- `init`: a parameter initializer. `act_fn`: an activation or `None`.

## Initializers

`from orchestra.initialization import ...`

| Class | Description |
|-------|-------------|
| `Kaiming()` | Kaiming / He normal. |
| `Xavier()` | Xavier / Glorot normal. |
| `Lecun()` | LeCun normal. |
| `XavierUniform()` | Xavier uniform. |
| `LecunUniform()` | LeCun uniform. |
| `Const(value)` | All parameters set to `value`. |
| `Uniform(low, high)` | Uniform on `[low, high)`. |
| `UniformInclusive(low, high)` | Uniform on `[low, high]`. |
| `Normal(mean, std_dev)` | Gaussian. |

## Activations

`from orchestra.activations import ...`

| Class | Description |
|-------|-------------|
| `Sigmoid(amp=1.0)` | `amp / (1 + exp(-x))`. |
| `Tanh(amp=1.0)` | `amp * tanh(x)`. |
| `ReLU(slope)` | Leaky ReLU; `slope` (leakiness) must be in `[0.0, 1.0]` — pass `0.0` for plain ReLU. |
| `Softmax()` | Softmax over the layer output. |

## Datasets

`from orchestra.datasets import ...`

| Class | Signature | Notes |
|-------|-----------|-------|
| `InlineDataset` | `InlineDataset(samples, labels, x_size, y_size)` | `samples`/`labels` are flat, row-major `list[float]`. |
| `LocalDataset` | `LocalDataset(samples_path, labels_path, x_size, y_size)` | Files hold raw little-endian packed `f32` values; both must exist. |

`x_size`/`y_size` are the input/output feature counts per sample (both `> 0`).

## Optimizers

`from orchestra.optimizers import ...`

| Class | Signature | Notes |
|-------|-----------|-------|
| `GradientDescent` | `GradientDescent(lr)` | `lr > 0`. |
| `GradientDescentWithMomentum` | `GradientDescentWithMomentum(lr, mu)` | `lr > 0`, momentum `mu` in `[0.0, 1.0]`. |
| `Adam` | `Adam(lr, b1=0.9, b2=0.999, eps=1e-8)` | `lr`, `eps > 0`; `b1`, `b2` in `[0.0, 1.0]`. |

## Loss functions

`from orchestra.loss_fns import ...`

| Class | Use |
|-------|-----|
| `Mse()` | Regression. |
| `CrossEntropy()` | Classification. |

## Synchronization, stores, serializers

Used by `parameter_server` and `strategy_switch` (sync + store), and by all strategies
(serializer):

| Group | Classes |
|-------|---------|
| Sync (`orchestra.sync`) | `BarrierSync()` — workers sync each round (consistent). `NonBlockingSync()` — no waiting (higher throughput). |
| Store (`orchestra.store`) | `BlockingStore()` — updates under a lock. `WildStore()` — lock-free, faster, may race. |
| Serializer (`orchestra.serializer`) | `BaseSerializer()` — dense gradients (default). `SparseSerializer(r)` — compresses gradients, `r` in `[0.0, 1.0]`. |

---

## Training strategies

All three builders return an opaque `PyTrainingConfig` for `orchestrate(...)`. They share
these optional arguments: `serializer=None` (defaults to `BaseSerializer()`),
`offline_epochs=0` (extra local epochs per sync round), `seed=None`, and
`early_stopping_tolerance=None` (see below).

### `all_reduce(...)`

Collective gradient averaging — no parameter server. Every node in `addrs` is a worker.

```python
training = all_reduce(
    addrs=["node-0:40000", "node-1:40001"],
    dataset=dataset,
    optimizer=GradientDescent(lr=0.01),
    loss_fn=Mse(),
    max_epochs=1000,
    batch_size=32,
)
```

### `parameter_server(...)`

`nservers` nodes serve parameters; the remaining `len(addrs) - nservers` are workers.

```python
training = parameter_server(
    addrs=["node-0:40000", "node-1:40001", "node-2:40002"],
    nservers=1,                 # 1 server, 2 workers
    dataset=dataset,
    optimizer=GradientDescent(lr=0.01),
    loss_fn=Mse(),
    sync=BarrierSync(),
    store=BlockingStore(),
    max_epochs=1000,
    batch_size=32,
)
```

### `strategy_switch(...)`

Starts as all-reduce and switches to parameter server once relative loss improvement drops
below an internal threshold. Same signature as `parameter_server` (`sync`/`store` apply to
the PS phase).

```python
training = strategy_switch(
    addrs=["node-0:40000", "node-1:40001", "node-2:40002"],
    nservers=1,
    dataset=dataset,
    optimizer=GradientDescent(lr=0.01),
    loss_fn=Mse(),
    sync=BarrierSync(),
    store=BlockingStore(),
    max_epochs=1000,
    batch_size=32,
)
```

---

## Orchestration and results

### `orchestrate(model, training) -> Session`

Connects to every node in `addrs` and starts training. Releases the GIL during the run.

### `Session`

- `session.wait() -> TrainedModel` — blocks until training completes (or stops early) and
  returns the trained model. Renders a live progress bar to stdout.
- `session.stop()` — requests an orderly stop at the next epoch boundary. Non-blocking;
  safe to call from another thread while `wait()` is running.

### `TrainedModel`

- `weights() -> list[float]` — final parameters as a flat vector (weights then biases,
  in layer order).
- `loss_history() -> list[float]` — average loss per completed epoch.
- `save_safetensors(path)` — saves the model in
  [safetensors](https://github.com/huggingface/safetensors) format.

## Early stopping

Pass `early_stopping_tolerance` to any strategy to stop automatically when the loss plateaus.
Training halts at the next epoch boundary once, between two consecutive sync rounds:

```
|prev_avg_loss - curr_avg_loss| < tolerance
```

- `None` (default) disables it; otherwise the tolerance must be `> 0`.
- The check runs **per sync round**, not per batch — with `offline_epochs > 0` a round
  spans several local epochs.

---

## End-to-end example (MNIST, parameter server)

Mirrors [`local.py`](./local.py) — the reference driver used by `run.sh`.

```python
import os

import orchestra
from orchestra import Sequential, orchestrate
from orchestra.arch import Dense
from orchestra.activations import Sigmoid
from orchestra.initialization import Kaiming
from orchestra.datasets import LocalDataset
from orchestra.optimizers import GradientDescent
from orchestra.loss_fns import Mse
from orchestra.sync import NonBlockingSync
from orchestra.store import WildStore
from orchestra.serializer import SparseSerializer

# One address per running `node` process. Roles are assigned at runtime.
addrs = [f"node-{i}:{40000 + i}" for i in range(int(os.environ["NODES"]))]
nservers = int(os.environ["SERVERS"])   # must be < len(addrs)

model = Sequential([
    Dense(128, Kaiming(), act_fn=Sigmoid()),
    Dense(64, Kaiming(), act_fn=Sigmoid()),
    Dense(10, Kaiming()),
])

training = orchestra.parameter_server(
    addrs=addrs,
    nservers=nservers,
    dataset=LocalDataset("data/mnist_samples.bin", "data/mnist_labels.bin",
                         x_size=784, y_size=10),
    optimizer=GradientDescent(lr=0.01),
    loss_fn=Mse(),
    sync=NonBlockingSync(),
    store=WildStore(),
    serializer=SparseSerializer(r=0.9),
    max_epochs=50,
    batch_size=256,
    offline_epochs=1,
    seed=42,
    early_stopping_tolerance=1e-4,
)

session = orchestrate(model, training)
trained = session.wait()
trained.save_safetensors("model.safetensors")
```

To bring up the nodes and run this driver, see [`run.sh`](./run.sh), which starts the
`node` processes via Docker and assigns server/worker roles at runtime.
