# orchestra-py

Python FFI for the Oxidized Neural Orchestra distributed training system. Exposes a high-level API to define models, configure distributed training sessions, and retrieve trained parameters.

## Installation

Build and install the wheel from the workspace root:

```bash
maturin develop --release
```

Or install directly into your environment:

```bash
pip install -e orchestra-py/
```

## Concepts

Training is split into three steps:

1. **Define the model** — a `Sequential` stack of `Dense` and/or `Conv2d` layers.
2. **Configure training** — pick an algorithm (`parameter_server` or `all_reduce`), dataset, optimizer, and hyperparameters.
3. **Orchestrate** — hand both to `orchestrate()`, which connects to the running workers and servers and returns a `Session`. Call `session.wait()` to block until training completes.

The system is **distributed**: workers and parameter servers run as separate processes (local or remote). The Python process acts as the orchestrator.

---

## Quick start

```python
from orchestra import Sequential, orchestrate, parameter_server
from orchestra._orchestra import (
    Dense, Kaiming, Sigmoid,
    GradientDescent, Mse,
    InlineDataset,
    BarrierSync, BlockingStore,
)

# XOR dataset — 4 samples, 2 inputs, 1 output
dataset = InlineDataset(
    samples=[0., 0., 0., 1., 1., 0., 1., 1.],
    labels= [0.,       1.,       1.,       0.],
    x_size=2,
    y_size=1,
)

model = Sequential([
    Dense(4, Kaiming(), act_fn=Sigmoid()),
    Dense(1, Kaiming()),
])

training = parameter_server(
    worker_addrs=["127.0.0.1:50000"],
    server_addrs=["127.0.0.1:40000"],
    dataset=dataset,
    optimizer=GradientDescent(lr=1.0),
    loss_fn=Mse(),
    sync=BarrierSync(),
    store=BlockingStore(),
    max_epochs=500,
    batch_size=4,
)

session = orchestrate(model, training)
trained = session.wait()

print(trained.weights())
trained.save_safetensors("xor.safetensors")
```

---

## Model definition

### `Sequential(layers)`

A sequential model. Layers are applied in order.

| Argument | Type | Description |
|----------|------|-------------|
| `layers` | `list[Dense \| Conv2d]` | At least one layer required. |

### `Dense(output_size, init, act_fn=None)`

A fully-connected dense layer.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `output_size` | `int` | — | Number of output neurons. Must be > 0. |
| `init` | initializer | — | Weight and bias initializer. |
| `act_fn` | activation or `None` | `None` | Optional activation applied after the linear transform. |

### `Conv2d(input_dim, kernel_dim, stride, padding, init, act_fn=None)`

A 2D convolutional layer. The kernel is square — `kernel_size` applies to both spatial dimensions.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `input_dim` | `tuple[int, int, int]` | — | Input shape as `(in_channels, height, width)`. All values must be > 0. |
| `kernel_dim` | `tuple[int, int, int]` | — | Kernel shape as `(filters, in_channels, kernel_size)`. All values must be > 0. |
| `stride` | `int` | — | Convolution stride applied to both spatial dimensions. Must be > 0. |
| `padding` | `int` | — | Zero-padding added to each spatial side of the input. |
| `init` | initializer | — | Weight and bias initializer. |
| `act_fn` | activation or `None` | `None` | Optional activation applied after the convolution. |

When mixing `Conv2d` and `Dense` layers in a `Sequential`, the output of the last convolutional layer is automatically flattened into a 2D tensor before the first dense layer.

---

## Initializers

All initializers are zero-argument unless noted.

| Class | Description |
|-------|-------------|
| `Kaiming()` | Kaiming / He — recommended for layers followed by ReLU-like activations. |
| `Xavier()` | Xavier / Glorot normal — good default for sigmoid activations. |
| `Lecun()` | LeCun normal. |
| `XavierUniform()` | Xavier uniform variant. |
| `LecunUniform()` | LeCun uniform variant. |
| `Const(value)` | All parameters set to `value`. |
| `Uniform(low, high)` | Sampled uniformly from `[low, high)`. |
| `UniformInclusive(low, high)` | Sampled uniformly from `[low, high]`. |
| `Normal(mean, std_dev)` | Sampled from a normal distribution. |

---

## Activations

| Class | Args | Description |
|-------|------|-------------|
| `Sigmoid(amp=1.0)` | `amp: float` | Sigmoid with configurable amplitude: `amp / (1 + exp(-x))`. |

---

## Datasets

### `InlineDataset(samples, labels, x_size, y_size)`

Defines a dataset directly in Python memory.

| Argument | Type | Description |
|----------|------|-------------|
| `samples` | `list[float]` | Flat row-major list of input features. |
| `labels` | `list[float]` | Flat row-major list of output targets. |
| `x_size` | `int` | Number of input features per sample. |
| `y_size` | `int` | Number of output values per sample. |

### `LocalDataset(samples_path, labels_path, x_size, y_size)`

Loads data from binary files of packed little-endian `f32` values. Also accepts `.csv` and `.tsv` files — they are converted to binary automatically on first run.

| Argument | Type | Description |
|----------|------|-------------|
| `samples_path` | `str` | Path to the samples file. |
| `labels_path` | `str` | Path to the labels file. |
| `x_size` | `int` | Number of input features per sample. |
| `y_size` | `int` | Number of output values per sample. |

---

## Optimizer

### `GradientDescent(lr)`

Vanilla gradient descent. `lr` is the learning rate (float, must be > 0).

---

## Loss functions

| Class | Description |
|-------|-------------|
| `Mse()` | Mean squared error. Use for regression. |
| `CrossEntropy()` | Cross-entropy. Use for classification. |

---

## Training algorithms

### `parameter_server(...)`

Distributed training via a central parameter server cluster. Workers push gradients to the servers after each sync round; servers apply updates and push the new parameters back.

```python
training = parameter_server(
    worker_addrs=["127.0.0.1:50000", "127.0.0.1:50001"],
    server_addrs=["127.0.0.1:40000"],
    dataset=dataset,
    optimizer=GradientDescent(lr=0.01),
    loss_fn=Mse(),
    sync=BarrierSync(),
    store=BlockingStore(),
    max_epochs=1000,
    batch_size=32,
    serializer=None,           # optional — BaseSerializer() by default
    offline_epochs=0,          # optional — extra local epochs per sync round
    seed=42,                   # optional — for reproducibility
    early_stopping_tolerance=None,  # optional — see Early Stopping section
)
```

**Synchronization strategies:**

| Class | Description |
|-------|-------------|
| `BarrierSync()` | All workers sync before the parameter server applies updates. Consistent convergence. |
| `NonBlockingSync()` | Workers proceed without waiting for each other. Higher throughput, less consistency. |

**Parameter store strategies:**

| Class | Description |
|-------|-------------|
| `BlockingStore()` | Gradient updates are applied under a lock. Safe under concurrent workers. |
| `WildStore()` | Gradient updates applied without locking. Faster, may cause race conditions. |

**Gradient serializers:**

| Class | Description |
|-------|-------------|
| `BaseSerializer()` | Gradients are always sent in full. Default. |
| `SparseSerializer(r=0.01)` | Only gradients above threshold `r` are sent. Reduces bandwidth at the cost of precision. `r` must be in `[0.0, 1.0]`. |

---

### `all_reduce(...)`

Distributed training via all-reduce collective — workers exchange gradients directly without a parameter server.

```python
training = all_reduce(
    worker_addrs=["127.0.0.1:50000", "127.0.0.1:50001"],
    dataset=dataset,
    optimizer=GradientDescent(lr=0.01),
    loss_fn=Mse(),
    max_epochs=1000,
    batch_size=32,
    serializer=None,
    offline_epochs=0,
    seed=42,
    early_stopping_tolerance=None,
)
```

> **Note:** All-reduce session finalization is not yet implemented. Training will start but the final model cannot be retrieved at the end of the session.

---

## Session

### `orchestrate(model, training) -> Session`

Connects to all workers and servers and starts training. Blocks the current thread during the connection phase; returns a `Session` handle once all nodes are ready.

```python
session = orchestrate(model, training)
```

### `session.wait() -> TrainedModel`

Blocks until training completes (or stops early) and returns the trained model. Releases the GIL while waiting — other Python threads can run concurrently.

```python
trained = session.wait()
```

### `session.stop()`

Requests an orderly stop of the training session at the next epoch boundary. Non-blocking — the stop signal is sent asynchronously. `wait()` will return once the current epoch finishes and the final parameters are collected.

```python
session.stop()
```

Can be called from another thread while `wait()` is blocking:

```python
import threading

session = orchestrate(model, training)

def stop_after(seconds):
    time.sleep(seconds)
    session.stop()

threading.Thread(target=stop_after, args=(30,), daemon=True).start()
trained = session.wait()
```

---

## Trained model

### `trained.weights() -> list[float]`

Returns the final parameters as a flat list in layer order: weights then biases for each layer.

### `trained.save_safetensors(path)`

Saves the model in [safetensors](https://github.com/huggingface/safetensors) format. Each layer produces two tensors:

- Dense: `layer_N.weight` — shape `[input_size, output_size]`, `layer_N.bias` — shape `[output_size]`
- Conv2d: `layer_N.weight` — shape `[filters, in_channels, kernel_size, kernel_size]`, `layer_N.bias` — shape `[filters]`

---

## Early stopping

Set `early_stopping_tolerance` in `parameter_server()` or `all_reduce()` to stop training automatically when the loss stops improving.

Training stops at the next epoch boundary when:

```
|prev_avg_loss - curr_avg_loss| < tolerance
```

where `prev_avg_loss` and `curr_avg_loss` are the average losses across all workers for two consecutive sync rounds.

- If `tolerance` is `None` (default), early stopping is disabled.
- `tolerance` must be strictly greater than 0.
- The check runs **per sync round**, not per batch. With `offline_epochs > 0`, a sync round covers multiple local epochs.
- Stopping always happens at an epoch boundary — the system never interrupts mid-epoch.

```python
training = parameter_server(
    ...
    max_epochs=5000,
    early_stopping_tolerance=1e-4,
)
```

See the [early stopping example](#example-early-stopping-with-parameter-server) below for a full runnable script.

---

## Examples

### Example: XOR with parameter server

```python
from orchestra import Sequential, orchestrate, parameter_server
from orchestra._orchestra import (
    Dense, Kaiming, Sigmoid,
    GradientDescent, Mse,
    InlineDataset,
    BarrierSync, BlockingStore,
)

dataset = InlineDataset(
    samples=[0., 0., 0., 1., 1., 0., 1., 1.],
    labels= [0.,       1.,       1.,       0.],
    x_size=2,
    y_size=1,
)

model = Sequential([
    Dense(4, Kaiming(), act_fn=Sigmoid()),
    Dense(1, Kaiming()),
])

training = parameter_server(
    worker_addrs=["127.0.0.1:50000"],
    server_addrs=["127.0.0.1:40000"],
    dataset=dataset,
    optimizer=GradientDescent(lr=1.0),
    loss_fn=Mse(),
    sync=BarrierSync(),
    store=BlockingStore(),
    max_epochs=500,
    batch_size=4,
    seed=42,
)

session = orchestrate(model, training)
trained = session.wait()
trained.save_safetensors("xor.safetensors")
```

---

### Example: early stopping with parameter server

```python
from orchestra import Sequential, orchestrate, parameter_server
from orchestra._orchestra import (
    Dense, Kaiming, Sigmoid,
    GradientDescent, Mse,
    InlineDataset,
    BarrierSync, BlockingStore,
)

dataset = InlineDataset(
    samples=[0., 0., 0., 1., 1., 0., 1., 1.],
    labels= [0.,       1.,       1.,       0.],
    x_size=2,
    y_size=1,
)

model = Sequential([
    Dense(8, Kaiming(), act_fn=Sigmoid()),
    Dense(4, Kaiming(), act_fn=Sigmoid()),
    Dense(1, Kaiming()),
])

training = parameter_server(
    worker_addrs=["127.0.0.1:50000"],
    server_addrs=["127.0.0.1:40000"],
    dataset=dataset,
    optimizer=GradientDescent(lr=1.0),
    loss_fn=Mse(),
    sync=BarrierSync(),
    store=BlockingStore(),
    max_epochs=5000,           # high ceiling — early stopping takes over
    batch_size=4,
    seed=42,
    early_stopping_tolerance=1e-4,
)

session = orchestrate(model, training)
trained = session.wait()
trained.save_safetensors("xor_early_stop.safetensors")
```

---

### Example: image classification with Conv2d

```python
from orchestra import Sequential, orchestrate, parameter_server
from orchestra._orchestra import (
    Conv2d, Dense, Kaiming, Sigmoid,
    GradientDescent, CrossEntropy,
    LocalDataset,
    BarrierSync, BlockingStore,
)

# Dataset of 28x28 grayscale images, 10 classes
dataset = LocalDataset(
    samples_path="data/images.bin",
    labels_path="data/labels.bin",
    x_size=784,   # 1 * 28 * 28
    y_size=10,
)

model = Sequential([
    Conv2d(
        input_dim=(1, 28, 28),
        kernel_dim=(32, 1, 3),
        stride=1,
        padding=1,
        init=Kaiming(),
        act_fn=Sigmoid(),
    ),
    Conv2d(
        input_dim=(32, 28, 28),
        kernel_dim=(64, 32, 3),
        stride=2,
        padding=1,
        init=Kaiming(),
    ),
    Dense(10, Kaiming()),
])

training = parameter_server(
    worker_addrs=["127.0.0.1:50000"],
    server_addrs=["127.0.0.1:40000"],
    dataset=dataset,
    optimizer=GradientDescent(lr=0.01),
    loss_fn=CrossEntropy(),
    sync=BarrierSync(),
    store=BlockingStore(),
    max_epochs=20,
    batch_size=64,
    seed=0,
)

session = orchestrate(model, training)
trained = session.wait()
trained.save_safetensors("image_clf.safetensors")
```

---

### Example: MNIST with local dataset and sparse gradients

```python
from orchestra import Sequential, orchestrate, parameter_server
from orchestra._orchestra import (
    Dense, Kaiming, Sigmoid,
    GradientDescent, CrossEntropy,
    LocalDataset,
    BarrierSync, BlockingStore,
    SparseSerializer,
)

dataset = LocalDataset(
    samples_path="data/mnist_samples.bin",
    labels_path="data/mnist_labels.bin",
    x_size=784,
    y_size=10,
)

model = Sequential([
    Dense(256, Kaiming(), act_fn=Sigmoid()),
    Dense(128, Kaiming(), act_fn=Sigmoid()),
    Dense(10,  Kaiming()),
])

training = parameter_server(
    worker_addrs=["127.0.0.1:50000", "127.0.0.1:50001"],
    server_addrs=["127.0.0.1:40000", "127.0.0.1:40001"],
    dataset=dataset,
    optimizer=GradientDescent(lr=0.05),
    loss_fn=CrossEntropy(),
    sync=BarrierSync(),
    store=BlockingStore(),
    max_epochs=50,
    batch_size=256,
    serializer=SparseSerializer(r=0.01),
    seed=0,
)

session = orchestrate(model, training)
trained = session.wait()
trained.save_safetensors("mnist.safetensors")
```

---

### Example: manual stop from another thread

```python
import threading
import time
from orchestra import Sequential, orchestrate, parameter_server
from orchestra._orchestra import (
    Dense, Xavier,
    GradientDescent, Mse,
    InlineDataset,
    BarrierSync, BlockingStore,
)

dataset = InlineDataset(
    samples=[0., 0., 0., 1., 1., 0., 1., 1.],
    labels= [0.,       1.,       1.,       0.],
    x_size=2,
    y_size=1,
)

model = Sequential([Dense(4, Xavier()), Dense(1, Xavier())])

training = parameter_server(
    worker_addrs=["127.0.0.1:50000"],
    server_addrs=["127.0.0.1:40000"],
    dataset=dataset,
    optimizer=GradientDescent(lr=0.5),
    loss_fn=Mse(),
    sync=BarrierSync(),
    store=BlockingStore(),
    max_epochs=10000,
    batch_size=4,
)

session = orchestrate(model, training)

def stop_after(seconds):
    time.sleep(seconds)
    print("requesting stop...")
    session.stop()

threading.Thread(target=stop_after, args=(10,), daemon=True).start()

trained = session.wait()
trained.save_safetensors("timed_stop.safetensors")
```
