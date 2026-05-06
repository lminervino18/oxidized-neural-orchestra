# orchestui — Config Reference

The TUI reads two JSON config files at startup: `model.json` and `training.json`.

## Running

```bash
cargo run -p orchestui
```

The TUI will prompt for the paths to both config files. Leave a field blank to use `model.json` / `training.json` from the current directory. Press `?` at either prompt to see an inline config example.

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
| `"softmax"` | — | Softmax over the output vector. |

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
  "seed":           42,
  "serializer":     "base",
  "early_stopping": { "tolerance": 1e-4 }
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
| `serializer` | string or object | ❌ | Gradient serializer. Defaults to `"base"`. See below. |
| `early_stopping` | object | ❌ | Stop training when loss improvement falls below a threshold. Omit to disable. |

### Early stopping

```json
"early_stopping": { "tolerance": 1e-4 }
```

Training stops at the next epoch boundary when `|prev_avg_loss - curr_avg_loss| < tolerance`. `tolerance` must be strictly greater than 0.

---

### Algorithm (`algorithm`)

Two options:

#### `parameter_server`

Workers push gradients to centralized servers, which apply updates and return new parameters. Model parameters are sharded across servers.

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

##### Synchronizers

| Value | Description |
|---|---|
| `"barrier"` | All workers synchronize gradients at the end of each epoch before proceeding. Ensures consistent updates. |
| `"non_blocking"` | Workers send gradients and continue without waiting for others. Higher throughput, less consistency. |

##### Stores

| Value | Description |
|---|---|
| `"blocking"` | Parameter reads block until the latest update is applied. Consistent reads. |
| `"wild"` | Parameter reads may return stale values. Better throughput under high concurrency. |

#### `all_reduce`

Workers exchange and reduce gradients directly with each other — no parameter server required. Each worker ends up with the same averaged gradient and applies it locally.

```json
"algorithm": "all_reduce"
```

No additional fields. Workers are identified solely by `worker_addrs`.

---

### Dataset (`dataset`)

| Field | Type | Required | Description |
|---|---|---|---|
| `src` | object | ✅ | Data source. Either `local` or `inline`. |
| `x_size` | integer ≥ 1 | ✅ | Number of input features per sample. |
| `y_size` | integer ≥ 1 | ✅ | Number of output values per sample. |

#### `local` source

Reads from binary files of packed little-endian `f32` values. `.csv` and `.tsv` files are converted to binary automatically on first run.

```json
"src": {
  "local": {
    "samples_path": "data/mnist_train_samples.bin",
    "labels_path":  "data/mnist_train_labels.bin"
  }
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `samples_path` | string | ✅ | Path to the samples file, relative to the working directory where the TUI is launched. |
| `labels_path` | string | ✅ | Path to the labels file, relative to the working directory where the TUI is launched. |

#### `inline` source

Embeds the dataset directly in the config.

```json
"src": {
  "inline": {
    "samples": [1.0, 2.0, 3.0, 4.0],
    "labels":  [2.0, 4.0, 6.0, 8.0]
  }
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `samples` | `[f32, ...]` | ✅ | Flat row-major array of input features. Length must be divisible by `x_size`. |
| `labels` | `[f32, ...]` | ✅ | Flat row-major array of output targets. Length must be divisible by `y_size`. |

---

### Optimizer (`optimizer`)

#### `gradient_descent`

Vanilla stochastic gradient descent.

```json
"optimizer": { "gradient_descent": { "lr": 0.01 } }
```

| Field | Type | Required | Description |
|---|---|---|---|
| `lr` | f32 | ✅ | Learning rate. Must be > 0. |

#### `gradient_descent_with_momentum`

Gradient descent with momentum.

```json
"optimizer": { "gradient_descent_with_momentum": { "lr": 0.01, "mu": 0.9 } }
```

| Field | Type | Required | Description |
|---|---|---|---|
| `lr` | f32 | ✅ | Learning rate. Must be > 0. |
| `mu` | f32 | ✅ | Momentum coefficient. Must be in `[0, 1]`. |

#### `adam`

Adam optimizer.

```json
"optimizer": { "adam": { "lr": 0.001, "b1": 0.9, "b2": 0.999, "eps": 1e-8 } }
```

| Field | Type | Required | Description |
|---|---|---|---|
| `lr` | f32 | ✅ | Learning rate. Must be > 0. |
| `b1` | f32 | ✅ | First moment decay. Must be in `[0, 1]`. |
| `b2` | f32 | ✅ | Second moment decay. Must be in `[0, 1]`. |
| `eps` | f32 | ✅ | Numerical stability constant. Must be > 0. |

---

### Loss functions (`loss_fn`)

| Value | Description |
|---|---|
| `"mse"` | Mean squared error. Use for regression. |
| `"cross_entropy"` | Cross-entropy. Use for classification. |

---

### Serializer (`serializer`)

| Value | Description |
|---|---|
| `"base"` | Gradients are always sent in full. Default. |
| `{ "sparse_capable": { "r": 0.95 } }` | Only gradients above threshold `r` are sent. `r` must be in `[0.0, 1.0]`. |
