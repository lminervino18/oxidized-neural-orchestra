# orchestui — Config Reference

The TUI reads two JSON config files at startup: `model.json` and `training.json`.
Neither file is committed — copy the provided `.example.json` files as a starting point:

```bash
cp orchestui/model.example.json model.json
cp orchestui/training.example.json training.json
```

Then point the TUI at them when prompted, or place them anywhere and enter the paths at runtime.

---

## model.json

Defines the neural network architecture.

```json
{
  "layers": [ <layer>, ... ]
}
```

`layers` is required and must contain at least one entry.

### Layer types

#### `dense`

A fully-connected layer.

| Field | Type | Required | Description |
|---|---|---|---|
| `output_size` | integer ≥ 1 | ✅ | Number of output neurons. |
| `init` | initializer | ✅ | Weight initialization strategy. See below. |
| `act_fn` | activation | ❌ | Activation function applied after the linear transform. Omit for no activation. |

```json
{ "dense": { "output_size": 8, "init": "kaiming", "act_fn": { "sigmoid": { "amp": 1.0 } } } }
```

#### `conv`

A 2D convolutional layer. The kernel is square — `kernel_size` applies to both height and width.

| Field | Type | Required | Description |
|---|---|---|---|
| `input_dim` | `[integer, integer, integer]` | ✅ | Input shape as `[in_channels, height, width]`. |
| `kernel_dim` | `[integer, integer, integer]` | ✅ | Kernel shape as `[filters, in_channels, kernel_size]`. |
| `stride` | integer ≥ 1 | ✅ | Convolution stride applied to both spatial dimensions. |
| `padding` | integer ≥ 0 | ✅ | Zero-padding added to each spatial side of the input. |
| `init` | initializer | ✅ | Weight initialization strategy. See below. |
| `act_fn` | activation | ❌ | Activation function applied after the convolution. Omit for no activation. |

```json
{ "conv": { "input_dim": [1, 28, 28], "kernel_dim": [32, 1, 3], "stride": 1, "padding": 1, "init": "kaiming" } }
```

When combining convolutional and dense layers, the output of the last `conv` layer is automatically flattened before the first `dense` layer.

---

### Initializers (`init`)

| Value | Fields | Description |
|---|---|---|
| `"kaiming"` | — | Kaiming (He) normal. Recommended for layers followed by ReLU-like activations. |
| `"xavier"` | — | Xavier (Glorot) normal. Recommended for symmetric activations like sigmoid/tanh. |
| `"lecun"` | — | LeCun normal. |
| `"xavier_uniform"` | — | Xavier uniform variant. |
| `"lecun_uniform"` | — | LeCun uniform variant. |
| `{ "const": { "value": 0.0 } }` | `value: f32` | Initializes all parameters to a constant value. |
| `{ "uniform": { "low": -0.1, "high": 0.1 } }` | `low: f32`, `high: f32` | Uniform random in `[low, high)`. |
| `{ "uniform_inclusive": { "low": -0.1, "high": 0.1 } }` | `low: f32`, `high: f32` | Uniform random in `[low, high]`. |
| `{ "normal": { "mean": 0.0, "std_dev": 0.1 } }` | `mean: f32`, `std_dev: f32` | Gaussian random. |

---

### Activation functions (`act_fn`)

| Value | Fields | Description |
|---|---|---|
| `{ "sigmoid": { "amp": 1.0 } }` | `amp: f32` | Sigmoid: `amp / (1 + e^(-x))`. Use `amp: 1.0` for the standard sigmoid. |

---

## training.json

Defines the distributed training setup.

```json
{
  "worker_addrs":   [ <addr>, ... ],
  "algorithm":      { ... },
  "dataset":        { ... },
  "optimizer":      { ... },
  "loss_fn":        "mse",
  "batch_size":     4,
  "max_epochs":     500,
  "offline_epochs": 0,
  "seed":           42
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `worker_addrs` | `["host:port", ...]` | ✅ | Socket addresses of all worker processes. At least one required. |
| `algorithm` | object | ✅ | Distributed training algorithm. See below. |
| `dataset` | object | ✅ | Dataset source and shape. See below. |
| `optimizer` | object | ✅ | Optimizer to use during training. See below. |
| `loss_fn` | string | ✅ | Loss function. See below. |
| `batch_size` | integer ≥ 1 | ✅ | Mini-batch size. Must not exceed total number of samples. |
| `max_epochs` | integer ≥ 1 | ✅ | Maximum number of training epochs. |
| `offline_epochs` | integer ≥ 0 | ✅ | Extra local epochs each worker runs before syncing with servers. Use `0` to disable. |
| `seed` | integer | ❌ | Optional random seed for reproducibility. Omit for non-deterministic runs. |

---

### Algorithm (`algorithm`)

Only `parameter_server` is currently implemented.

#### `parameter_server`

| Field | Type | Required | Description |
|---|---|---|---|
| `server_addrs` | `["host:port", ...]` | ✅ | Socket addresses of all parameter server processes. At least one required. |
| `synchronizer` | string | ✅ | Gradient synchronization strategy. See below. |
| `store` | string | ✅ | Parameter store strategy. See below. |

```json
"algorithm": {
  "parameter_server": {
    "server_addrs": ["127.0.0.1:40000", "127.0.0.1:40001"],
    "synchronizer": "barrier",
    "store": "blocking"
  }
}
```

#### Synchronizers

| Value | Description |
|---|---|
| `"barrier"` | All workers synchronize gradients at the end of each epoch before proceeding. Ensures consistent updates. |
| `"non_blocking"` | Workers send gradients and continue without waiting for others. Higher throughput, less consistency. |

#### Stores

| Value | Description |
|---|---|
| `"blocking"` | Parameter reads block until the latest update is applied. Consistent reads. |
| `"wild"` | Parameter reads may return stale values. Better throughput under high concurrency. |

---

### Dataset (`dataset`)

| Field | Type | Required | Description |
|---|---|---|---|
| `src` | object | ✅ | Data source. Either `local` or `inline`. |
| `x_size` | integer ≥ 1 | ✅ | Number of input features per sample. |
| `y_size` | integer ≥ 1 | ✅ | Number of output values per sample. |

Each row in the dataset must contain exactly `x_size + y_size` contiguous `f32` values.

#### `local` source

Reads from a binary file of packed little-endian `f32` values.

```json
"src": { "local": { "path": "data/dataset" } }
```

| Field | Type | Required | Description |
|---|---|---|---|
| `path` | string | ✅ | Path to the binary dataset file, relative to the working directory where the TUI is launched. |

#### `inline` source

Embeds the dataset directly in the config as a flat array of `f32`.

```json
"src": { "inline": { "data": [1.0, 2.0, 2.0, 4.0, 3.0, 6.0] } }
```

| Field | Type | Required | Description |
|---|---|---|---|
| `data` | `[f32, ...]` | ✅ | Flat array of floats in row-major order. Length must be divisible by `x_size + y_size`. |

---

### Optimizer (`optimizer`)

#### `gradient_descent`

Vanilla stochastic gradient descent.

```json
"optimizer": { "gradient_descent": { "lr": 0.01 } }
```

| Field | Type | Required | Description |
|---|---|---|---|
| `lr` | f32 | ✅ | Learning rate. |

---

### Loss functions (`loss_fn`)

| Value | Description |
|---|---|
| `"mse"` | Mean squared error. |