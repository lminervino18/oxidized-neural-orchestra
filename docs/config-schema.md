# ONO Config Schema

The single source of truth for the two JSON files that drive a run:

- **`model.json`** — the neural-network architecture (`ModelConfig`).
- **`training.json`** — the distributed-training setup (`TrainingConfig`).

Both are parsed verbatim by the `orchestrator` crate (`orchestrator::configs`),
so this reference is derived directly from the serde types in
[`orchestrator/src/configs/model.rs`](../orchestrator/src/configs/model.rs) and
[`orchestrator/src/configs/training.rs`](../orchestrator/src/configs/training.rs).
Working examples ship with `orchestui`:
[`model.example`](../orchestui/model.example) and
[`training.example`](../orchestui/training.example).

> **Enum encoding.** Rust enums are serialized in `snake_case`. A **unit
> variant** (no fields) is a bare string, e.g. `"kaiming"`. A **struct variant**
> (with fields) is a single-key object, e.g. `{ "sigmoid": { "amp": 1.0 } }`.

---

## `model.json`

```jsonc
{
  "layers": [ <Layer>, <Layer>, ... ]   // required, at least one
}
```

`layers` run in order. The output of the last `conv` / `max_pooling` layer is
flattened automatically before the first `dense` layer.

### Minimal valid example

```json
{
  "layers": [
    { "dense": { "output_size": 8, "init": "kaiming", "act_fn": { "sigmoid": { "amp": 1.0 } } } },
    { "dense": { "output_size": 4, "init": "kaiming", "act_fn": { "sigmoid": { "amp": 1.0 } } } },
    { "dense": { "output_size": 1, "init": "kaiming" } }
  ]
}
```

### Layer types

Each entry in `layers` is a single-key object keyed by the layer type.

#### `dense` — fully-connected layer

| Field | Type | Required | Description |
|---|---|:---:|---|
| `output_size` | integer ≥ 1 | ✅ | Number of output neurons. |
| `init` | [initializer](#initializers-init) | ✅ | Weight initialization strategy. |
| `act_fn` | [activation](#activation-functions-act_fn) | ❌ | Activation applied after the linear transform. Omit / `null` for none. |

#### `conv` — 2D convolutional layer (square kernel)

| Field | Type | Required | Description |
|---|---|:---:|---|
| `input_dim` | `[int≥1, int≥1, int≥1]` | ✅ | Input shape `[in_channels, height, width]`. |
| `kernel_dim` | `[int≥1, int≥1, int≥1]` | ✅ | Kernel shape `[filters, in_channels, kernel_size]`. |
| `stride` | integer ≥ 1 | ✅ | Stride, applied to both spatial dimensions. |
| `padding` | integer ≥ 0 | ✅ | Zero-padding added to each spatial side. |
| `init` | [initializer](#initializers-init) | ✅ | Weight initialization strategy. |
| `act_fn` | [activation](#activation-functions-act_fn) | ❌ | Activation applied after the convolution. Omit for none. |

```json
{ "conv": { "input_dim": [1, 28, 28], "kernel_dim": [32, 1, 3], "stride": 1, "padding": 1, "init": "kaiming" } }
```

#### `max_pooling` — 2D max-pooling layer (square filter)

| Field | Type | Required | Description |
|---|---|:---:|---|
| `input_dim` | `[int≥1, int≥1, int≥1]` | ✅ | Input shape `[in_channels, height, width]`. |
| `filter_size` | integer ≥ 1 | ✅ | Square pooling window size. |
| `stride` | integer ≥ 1 | ✅ | Pooling stride. |
| `padding` | integer ≥ 0 | ✅ | Zero-padding added to each spatial side. |
| `act_fn` | [activation](#activation-functions-act_fn) | ❌ | Activation applied after pooling. Omit for none. |

```json
{ "max_pooling": { "input_dim": [32, 28, 28], "filter_size": 2, "stride": 2, "padding": 0 } }
```

### Initializers (`init`)

| Value | Fields | Description |
|---|---|---|
| `"kaiming"` | — | Kaiming (He) normal. For ReLU-like activations. |
| `"xavier"` | — | Xavier (Glorot) normal. For symmetric activations (sigmoid/tanh). |
| `"lecun"` | — | LeCun normal. |
| `"xavier_uniform"` | — | Xavier uniform variant. |
| `"lecun_uniform"` | — | LeCun uniform variant. |
| `{ "const": { "value": 0.0 } }` | `value: f32` | All parameters set to a constant. |
| `{ "uniform": { "low": -0.1, "high": 0.1 } }` | `low, high: f32` | Uniform in `[low, high)`. |
| `{ "uniform_inclusive": { "low": -0.1, "high": 0.1 } }` | `low, high: f32` | Uniform in `[low, high]`. |
| `{ "normal": { "mean": 0.0, "std_dev": 0.1 } }` | `mean, std_dev: f32` | Gaussian. |

### Activation functions (`act_fn`)

| Value | Fields | Description |
|---|---|---|
| `{ "sigmoid": { "amp": 1.0 } }` | `amp: f32` | `amp / (1 + e^(-x))`. Use `amp: 1.0` for the standard sigmoid. |
| `{ "tanh": { "amp": 1.0 } }` | `amp: f32` | `amp * tanh(x)`. |
| `{ "relu": { "slope": 0.0 } }` | `slope: 0.0..=1.0` | Leaky ReLU; `slope: 0.0` is standard ReLU. |
| `"softmax"` | — | Softmax over the output vector. |

---

## `training.json`

| Field | Type | Required | Description |
|---|---|:---:|---|
| `addrs` | `["host:port", ...]` | ✅ | Socket addresses of **all** participating nodes. Roles (worker vs. server) are assigned at runtime — one flat list, not separate worker/server lists. At least one required. |
| `algorithm` | string or object | ✅ | Distributed training [algorithm](#algorithm-algorithm). |
| `dataset` | object | ✅ | Dataset [source and shape](#dataset-dataset). |
| `optimizer` | object | ✅ | [Optimizer](#optimizer-optimizer). |
| `loss_fn` | string | ✅ | `"mse"` (regression) or `"cross_entropy"` (classification). |
| `batch_size` | integer ≥ 1 | ✅ | Mini-batch size. Must not exceed the number of samples. |
| `max_epochs` | integer ≥ 1 | ✅ | Maximum number of training epochs. |
| `offline_epochs` | integer ≥ 0 | ✅ | Extra local epochs each worker runs before syncing. `0` to disable. |
| `serializer` | string or object | ❌ | Gradient [serializer](#serializer-serializer). Defaults to `"base"`. |
| `seed` | integer ≥ 0 | ❌ | Random seed for reproducibility. Omit / `null` for non-deterministic runs. |
| `early_stopping` | object | ❌ | `{ "tolerance": <f32 ≥ 0> }`. Stops at the next epoch boundary when the change in average loss falls below `tolerance`. Omit / `null` to disable. |

### Algorithm (`algorithm`)

Three variants. **Node roles are assigned by the orchestrator at runtime** — the
config never lists which address is a worker and which is a server. For the
server-based algorithms, `nservers` of the `addrs` are promoted to parameter
servers and the rest act as workers.

#### `all_reduce`

No parameter server. Workers reduce gradients directly with each other and each
applies the averaged result locally. Every node in `addrs` is a worker.

```json
"algorithm": "all_reduce"
```

#### `parameter_server`

Workers push gradients to centralized servers, which apply updates and return new
parameters. Parameters are sharded across the servers.

```json
"algorithm": {
  "parameter_server": { "nservers": 1, "synchronizer": "barrier", "store": "blocking" }
}
```

| Field | Type | Required | Description |
|---|---|:---:|---|
| `nservers` | integer ≥ 1 | ✅ | How many of the `addrs` become parameter servers. Must be fewer than `addrs.len()`. |
| `synchronizer` | string | ✅ | [`"barrier"` or `"non_blocking"`](#synchronizers-synchronizer). |
| `store` | string | ✅ | [`"blocking"` or `"wild"`](#stores-store). |

#### `strategy_switch`

Starts as all-reduce and, mid-run, upgrades `nservers` of the workers into
parameter servers, switching to the parameter-server strategy. Same fields as
`parameter_server`.

```json
"algorithm": {
  "strategy_switch": { "nservers": 1, "synchronizer": "non_blocking", "store": "wild" }
}
```

##### Synchronizers (`synchronizer`)

| Value | Description |
|---|---|
| `"barrier"` | Workers synchronize gradients each round before proceeding. Consistent updates. |
| `"non_blocking"` | Workers send gradients and continue without waiting. Higher throughput, less consistency. |

##### Stores (`store`)

| Value | Description |
|---|---|
| `"blocking"` | Parameter reads block until the latest update is applied. Consistent reads. |
| `"wild"` | Reads may return stale values. Better throughput under high concurrency. |

### Dataset (`dataset`)

| Field | Type | Required | Description |
|---|---|:---:|---|
| `src` | object | ✅ | Data source: `local` or `inline`. |
| `x_size` | integer ≥ 1 | ✅ | Number of input features per sample. |
| `y_size` | integer ≥ 1 | ✅ | Number of output values per sample. |

**`local`** — reads packed little-endian `f32` binary files. `.csv` / `.tsv`
files are converted to binary automatically on first run. Paths are relative to
the working directory where the run is launched.

```json
"src": { "local": { "samples_path": "data/samples.csv", "labels_path": "data/labels.csv" } }
```

**`inline`** — embeds the dataset directly in the config. `samples` length must be
divisible by `x_size`, `labels` by `y_size`.

```json
"src": { "inline": { "samples": [1.0, 2.0, 3.0, 4.0], "labels": [2.0, 4.0, 6.0, 8.0] } }
```

### Optimizer (`optimizer`)

| Variant | Fields | Notes |
|---|---|---|
| `gradient_descent` | `lr` (> 0) | Vanilla SGD. |
| `gradient_descent_with_momentum` | `lr` (> 0), `mu` (`0..=1`) | SGD with momentum. |
| `adam` | `lr` (> 0), `b1` (`0..=1`), `b2` (`0..=1`), `eps` (> 0) | Adam. |

```json
"optimizer": { "gradient_descent": { "lr": 0.01 } }
"optimizer": { "gradient_descent_with_momentum": { "lr": 0.01, "mu": 0.9 } }
"optimizer": { "adam": { "lr": 0.001, "b1": 0.9, "b2": 0.999, "eps": 1e-8 } }
```

### Serializer (`serializer`)

| Value | Description |
|---|---|
| `"base"` | Gradients are always sent in full. Default. |
| `{ "sparse_capable": { "r": 0.95 } }` | Only gradients above threshold `r` (`0.0..=1.0`) are sent. |

---

## Full `training.json` examples

### Parameter Server (inline dataset)

```json
{
  "addrs": ["node-0:40000", "node-1:40001", "node-2:40002"],
  "algorithm": {
    "parameter_server": { "nservers": 1, "synchronizer": "barrier", "store": "blocking" }
  },
  "dataset": {
    "src": { "inline": { "samples": [1.0, 2.0, 3.0, 4.0], "labels": [2.0, 4.0, 6.0, 8.0] } },
    "x_size": 1,
    "y_size": 1
  },
  "optimizer": { "gradient_descent": { "lr": 0.01 } },
  "loss_fn": "mse",
  "batch_size": 4,
  "max_epochs": 500,
  "offline_epochs": 0,
  "seed": 42,
  "serializer": "base",
  "early_stopping": { "tolerance": 1e-4 }
}
```

### All-Reduce (local dataset, minimal)

```json
{
  "addrs": ["node-0:40000", "node-1:40001", "node-2:40002"],
  "algorithm": "all_reduce",
  "dataset": {
    "src": { "local": { "samples_path": "data/samples.csv", "labels_path": "data/labels.csv" } },
    "x_size": 2,
    "y_size": 1
  },
  "optimizer": { "gradient_descent": { "lr": 0.01 } },
  "loss_fn": "mse",
  "batch_size": 4,
  "max_epochs": 500,
  "offline_epochs": 0
}
```

### Strategy Switch

```json
{
  "addrs": ["node-0:40000", "node-1:40001", "node-2:40002"],
  "algorithm": {
    "strategy_switch": { "nservers": 1, "synchronizer": "non_blocking", "store": "wild" }
  },
  "dataset": {
    "src": { "local": { "samples_path": "data/samples.csv", "labels_path": "data/labels.csv" } },
    "x_size": 2,
    "y_size": 1
  },
  "optimizer": { "adam": { "lr": 0.001, "b1": 0.9, "b2": 0.999, "eps": 1e-8 } },
  "loss_fn": "cross_entropy",
  "batch_size": 32,
  "max_epochs": 500,
  "offline_epochs": 1
}
```
