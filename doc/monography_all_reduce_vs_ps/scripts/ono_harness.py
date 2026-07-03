"""Self-contained experiment harness for the AllReduce vs Parameter-Server monography.

This module is INTENTIONALLY self-contained: it does NOT import anything from the
repo's ``benchmarks/`` package. The model/dataclass definitions, orchestra model
builders, the PyTorch baseline trainer, the Docker lifecycle and the derived
metrics are replicated here from the proven ``benchmarks/issue/*`` implementation
so the monography can be reproduced from a single directory.

Layout (all under doc/monography_all_reduce_vs_ps/):
    scripts/   this file + the two CLIs
    build/     generated compose.yaml (docker project ``ono-mono``)
    datasets/  mnist/ (and optionally emnist/) .bin files
    results/   raw/ jsonl + data/ subsets + artifacts/ + docker_logs/
    logs/      human-readable per-campaign logs

The orchestrator runs on the HOST (via the ``orchestra`` binding) and connects to
``node-i:PORT`` containers. ``/etc/hosts`` already maps node-0..node-6 to
127.0.0.1 and the compose publishes ``PORT:PORT`` so the addrs resolve.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import time
import traceback
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPTS_DIR = Path(__file__).resolve().parent
MONO_DIR = SCRIPTS_DIR.parent                       # doc/monography_all_reduce_vs_ps
REPO_ROOT = SCRIPTS_DIR.parents[2]                  # repo root

BUILD_DIR = MONO_DIR / "build"
COMPOSE_FILE = BUILD_DIR / "compose.yaml"
DATASETS_DIR = MONO_DIR / "datasets"
RESULTS_DIR = MONO_DIR / "results"
SUBSET_DIR = RESULTS_DIR / "data"
ARTIFACTS_DIR = RESULTS_DIR / "artifacts"
DOCKER_LOGS_DIR = RESULTS_DIR / "docker_logs"

# Source of the already-prepared full MNIST .bin files (float32, normalized,
# one-hot labels). download_datasets.py copies these into datasets/mnist/.
NB_MNIST_DIR = REPO_ROOT / "notebooks" / "mnist" / "data"

COMPOSE_PROJECT = "ono-mono"
# All node-i images are byte-identical (same Dockerfile, same MODE=release, same
# context) — only their runtime PORT/ports differ. So we build ONE shared image
# and every service references it. Building N images in parallel triggers N
# simultaneous Rust-release compilations, which OOM-kills low-RAM hosts.
NODE_IMAGE = "ono-node:release"
NODE_BASE_PORT = 40_000

X_SIZE = 784
Y_SIZE = 10
SEED = 42

# Mirrors the orchestrator's ConvergenceTracker window (adapter.rs uses 3).
EARLY_STOP_WINSIZE = 3

# Logged by a worker (RUST_LOG=info) when it switches from all-reduce to PS.
SWITCH_MARKER = "switching algorithm to parameter server"


# ── Typed primitives (replicated from benchmarks/issue/suites.py) ─────────────

class StrEnum(str, Enum):
    def __str__(self) -> str:
        return str(self.value)


class Activation(StrEnum):
    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    SOFTMAX = "softmax"


class Loss(StrEnum):
    CROSS_ENTROPY = "cross_entropy"
    MSE = "mse"


@dataclass(frozen=True)
class Conv:
    filters: int
    kernel: int
    stride: int = 1
    padding: int = 0
    act: Activation | None = None


@dataclass(frozen=True)
class MaxPool:
    size: int
    stride: int
    padding: int = 0


@dataclass(frozen=True)
class Dense:
    out: int
    act: Activation | None = None


@dataclass(frozen=True)
class TrainingConfig:
    lr: float
    batch_size: int
    max_epochs: int
    offline_epochs: int = 0
    early_stopping_tolerance: float | None = None
    subset: int | None = None
    eval: bool = False
    seed: int = SEED


@dataclass(frozen=True)
class Model:
    name: str
    input_dim: tuple[int, int, int]
    loss_fn: Loss
    layers: list = field(default_factory=list)
    reference: TrainingConfig | None = None


# ── Model registry ────────────────────────────────────────────────────────────

NIELSEN = Model(
    name="nielsen",
    input_dim=(1, 28, 28),
    loss_fn=Loss.CROSS_ENTROPY,
    layers=[
        Conv(filters=20, kernel=5, stride=1, padding=0),
        MaxPool(size=2, stride=2, padding=0),
        Dense(out=100, act=Activation.SIGMOID),
        Dense(out=Y_SIZE, act=Activation.SOFTMAX),
    ],
    reference=TrainingConfig(lr=0.1, batch_size=10, max_epochs=60),
)

LENET5 = Model(
    name="lenet5",
    input_dim=(1, 28, 28),
    loss_fn=Loss.CROSS_ENTROPY,
    layers=[
        Conv(filters=6, kernel=5, stride=1, padding=2, act=Activation.TANH),
        MaxPool(size=2, stride=2, padding=0),
        Conv(filters=16, kernel=5, stride=1, padding=0, act=Activation.TANH),
        MaxPool(size=2, stride=2, padding=0),
        Dense(out=120, act=Activation.TANH),
        Dense(out=84, act=Activation.TANH),
        Dense(out=Y_SIZE, act=Activation.SOFTMAX),
    ],
    reference=TrainingConfig(lr=0.05, batch_size=64, max_epochs=60),
)

# Pure-MLP family (no conv) — ReLU hidden activations, Softmax output, CE loss.
MLP_SMALL = Model(
    name="mlp_small",
    input_dim=(1, 28, 28),
    loss_fn=Loss.CROSS_ENTROPY,
    layers=[
        Dense(out=128, act=Activation.RELU),
        Dense(out=Y_SIZE, act=Activation.SOFTMAX),
    ],
    reference=TrainingConfig(lr=0.1, batch_size=64, max_epochs=40),
)

MLP_MEDIUM = Model(
    name="mlp_medium",
    input_dim=(1, 28, 28),
    loss_fn=Loss.CROSS_ENTROPY,
    layers=[
        Dense(out=512, act=Activation.RELU),
        Dense(out=512, act=Activation.RELU),
        Dense(out=Y_SIZE, act=Activation.SOFTMAX),
    ],
    reference=TrainingConfig(lr=0.1, batch_size=64, max_epochs=40),
)

MLP_LARGE = Model(
    name="mlp_large",
    input_dim=(1, 28, 28),
    loss_fn=Loss.CROSS_ENTROPY,
    layers=[
        Dense(out=1024, act=Activation.RELU),
        Dense(out=1024, act=Activation.RELU),
        Dense(out=Y_SIZE, act=Activation.SOFTMAX),
    ],
    reference=TrainingConfig(lr=0.1, batch_size=64, max_epochs=40),
)

MODELS = {m.name: m for m in (NIELSEN, LENET5, MLP_SMALL, MLP_MEDIUM, MLP_LARGE)}


def _resolve(model) -> Model:
    return MODELS[model] if isinstance(model, str) else model


# ── Dataset registry ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DatasetSpec:
    name: str
    x_size: int
    y_size: int
    source: str
    classes: str


DATASETS = {
    "mnist": DatasetSpec(
        name="mnist", x_size=784, y_size=10,
        source="LeCun MNIST (copied from notebooks/mnist/data, float32 [0,1])",
        classes="digits 0-9",
    ),
    # EMNIST digits split — same 10-class shape as MNIST, so the bundled models fit.
    "emnist": DatasetSpec(
        name="emnist", x_size=784, y_size=10,
        source="EMNIST digits split (NIST), 28x28 grayscale /255, one-hot",
        classes="digits 0-9",
    ),
    # FashionMNIST — Zalando drop-in for MNIST: same 28x28x1 / 10-class shape, but a
    # harder, less-homogeneous task (lower accuracy ceiling with simple nets).
    "fashion_mnist": DatasetSpec(
        name="fashion_mnist", x_size=784, y_size=10,
        source="Zalando FashionMNIST, 28x28 grayscale /255, one-hot",
        classes="garments 0-9",
    ),
}


def dataset_dir(name: str) -> Path:
    return DATASETS_DIR / name


def dataset_bins(name: str) -> dict:
    d = dataset_dir(name)
    return {
        "train_s": d / f"{name}_train_samples.bin",
        "train_l": d / f"{name}_train_labels.bin",
        "test_s": d / f"{name}_test_samples.bin",
        "test_l": d / f"{name}_test_labels.bin",
    }


def _sample_count(path: Path, x_size: int) -> int | None:
    try:
        return path.stat().st_size // (x_size * 4)
    except OSError:
        return None


def _copy_subset(src_s: Path, src_l: Path, dst_s: Path, dst_l: Path,
                 n: int, x_size: int, y_size: int) -> None:
    SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    dst_s.write_bytes(src_s.read_bytes()[: n * x_size * 4])
    dst_l.write_bytes(src_l.read_bytes()[: n * y_size * 4])


def resolve_dataset(name: str, subset: int | None = None):
    """Return (train_s, train_l, test_s, test_l, x_size, y_size).

    Raises FileNotFoundError if the dataset bins are missing — run
    download_datasets.py first.
    """
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; known: {list(DATASETS)}")
    spec = DATASETS[name]
    bins = dataset_bins(name)
    for key in ("train_s", "train_l", "test_s", "test_l"):
        if not bins[key].exists():
            raise FileNotFoundError(
                f"dataset {name!r} missing {bins[key]}; run download_datasets.py")

    train_s, train_l = bins["train_s"], bins["train_l"]
    if subset:
        full_n = _sample_count(bins["train_s"], spec.x_size) or 0
        if subset < full_n:
            sub_s = SUBSET_DIR / f"{name}_train_{subset}_samples.bin"
            sub_l = SUBSET_DIR / f"{name}_train_{subset}_labels.bin"
            if not (sub_s.exists() and sub_l.exists()):
                _copy_subset(bins["train_s"], bins["train_l"], sub_s, sub_l,
                             subset, spec.x_size, spec.y_size)
            train_s, train_l = sub_s, sub_l
    return train_s, train_l, bins["test_s"], bins["test_l"], spec.x_size, spec.y_size


# ── Layer dim propagation (replicated from benchmarks/issue/models.py) ────────

def _propagate(model: Model):
    """Yield (index, layer, in_dim, out_dim); dims are (C,H,W) or flat int."""
    c, h, w = model.input_dim
    flat = None
    for i, layer in enumerate(model.layers):
        if isinstance(layer, Conv):
            out_h = (h + 2 * layer.padding - layer.kernel) // layer.stride + 1
            out_w = (w + 2 * layer.padding - layer.kernel) // layer.stride + 1
            yield i, layer, (c, h, w), (layer.filters, out_h, out_w)
            c, h, w = layer.filters, out_h, out_w
        elif isinstance(layer, MaxPool):
            out_h = (h + 2 * layer.padding - layer.size) // layer.stride + 1
            out_w = (w + 2 * layer.padding - layer.size) // layer.stride + 1
            yield i, layer, (c, h, w), (c, out_h, out_w)
            h, w = out_h, out_w
        else:  # Dense
            in_flat = flat if flat is not None else c * h * w
            yield i, layer, in_flat, layer.out
            flat = layer.out


def param_count(model) -> int:
    """Trainable parameter count (weights + biases) for the models table."""
    model = _resolve(model)
    total = 0
    for _, layer, in_dim, out_dim in _propagate(model):
        if isinstance(layer, Conv):
            c = in_dim[0]
            total += layer.filters * (c * layer.kernel * layer.kernel) + layer.filters
        elif isinstance(layer, Dense):
            total += in_dim * out_dim + out_dim
        # MaxPool has no params
    return total


def model_summary(model) -> dict:
    model = _resolve(model)
    return {
        "name": model.name,
        "input_dim": list(model.input_dim),
        "loss_fn": str(model.loss_fn),
        "layers": [type(l).__name__ for l in model.layers],
        "params": param_count(model),
    }


# ── Orchestra model builder (replicated) ──────────────────────────────────────

def build_orchestra_model(model):
    from orchestra import Sequential
    from orchestra.activations import ReLU, Sigmoid, Softmax, Tanh
    from orchestra.arch import Conv2d
    from orchestra.arch import Dense as OrchDense
    from orchestra.arch import MaxPooling
    from orchestra.initialization import Xavier

    # orchestra's ReLU takes a leakiness slope (0.0 = standard ReLU).
    act_map = {Activation.SIGMOID: Sigmoid, Activation.TANH: Tanh,
               Activation.SOFTMAX: Softmax, Activation.RELU: lambda: ReLU(0.0)}

    def act(a):
        return act_map[a]() if a is not None else None

    model = _resolve(model)
    layers = []
    for _, layer, in_dim, _ in _propagate(model):
        if isinstance(layer, Conv):
            c, h, w = in_dim
            layers.append(Conv2d(input_dim=(c, h, w),
                                 kernel_dim=(layer.filters, c, layer.kernel),
                                 stride=layer.stride, padding=layer.padding,
                                 init=Xavier(), act_fn=act(layer.act)))
        elif isinstance(layer, MaxPool):
            layers.append(MaxPooling(input_dim=in_dim, filter_size=layer.size,
                                     stride=layer.stride, padding=layer.padding))
        else:
            layers.append(OrchDense(layer.out, Xavier(), act(layer.act)))
    return Sequential(layers)


# ── PyTorch reference + baseline trainer (replicated) ─────────────────────────

def _torch_act(activation):
    import torch
    return {
        Activation.SIGMOID: torch.sigmoid,
        Activation.TANH: torch.tanh,
        Activation.RELU: torch.relu,
        Activation.SOFTMAX: lambda x: torch.softmax(x, dim=-1),
    }.get(activation)


def _torch_modules(model: Model):
    import torch.nn as nn
    steps = []
    for _, layer, in_dim, out_dim in _propagate(model):
        if isinstance(layer, Conv):
            mod = nn.Conv2d(in_dim[0], layer.filters, layer.kernel,
                            stride=layer.stride, padding=layer.padding)
            steps.append(("conv", mod, layer.act))
        elif isinstance(layer, MaxPool):
            steps.append(("pool", nn.MaxPool2d(layer.size, stride=layer.stride), None))
        else:
            steps.append(("dense", nn.Linear(in_dim, out_dim), layer.act))
    return steps


def _build_net(model: Model, steps):
    import torch.nn as nn
    in_c, in_h, in_w = model.input_dim

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.mods = nn.ModuleList([m for _, m, _ in steps])

        def forward(self, x, logits=False):
            x = x.view(-1, in_c, in_h, in_w)
            flattened = False
            last = len(steps) - 1
            for i, (kind, mod, activation) in enumerate(steps):
                if kind == "dense" and not flattened:
                    x = x.flatten(start_dim=1)
                    flattened = True
                x = mod(x)
                if activation is not None and not (logits and i == last):
                    x = _torch_act(activation)(x)
            return x

    return Net()


def build_torch_ref(model, state_dict):
    """Load trained orchestra weights into a PyTorch module for accuracy eval."""
    import torch
    model = _resolve(model)
    steps = _torch_modules(model)
    for (kind, mod, _), (i, _, _, _) in zip(steps, _propagate(model)):
        if kind == "conv":
            with torch.no_grad():
                mod.weight.copy_(state_dict[f"layer_{i}.weight"])
                mod.bias.copy_(state_dict[f"layer_{i}.bias"])
        elif kind == "dense":
            with torch.no_grad():
                mod.weight.copy_(state_dict[f"layer_{i}.weight"].T)
                mod.bias.copy_(state_dict[f"layer_{i}.bias"])
    return _build_net(model, steps).eval()


def train_torch_baseline(model, train_x, train_y, test_x, test_y, cfg: TrainingConfig):
    """Single-process PyTorch training of the same architecture/recipe.

    Returns (loss_history, accuracy). CE is computed on logits (the final softmax
    is folded into the loss). Early stopping mirrors the orchestrator's tracker.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F

    model = _resolve(model)
    torch.manual_seed(cfg.seed)

    net = _build_net(model, _torch_modules(model)).train()
    opt = torch.optim.SGD(net.parameters(), lr=cfg.lr)

    xt = torch.tensor(np.asarray(train_x, dtype=np.float32))
    yt = torch.tensor(np.asarray(train_y, dtype=np.float32))
    targets = yt.argmax(dim=1)
    n = xt.shape[0]
    use_ce = model.loss_fn == Loss.CROSS_ENTROPY
    gen = torch.Generator().manual_seed(cfg.seed)

    tol = cfg.early_stopping_tolerance
    last, stable = None, 0

    loss_history = []
    for _ in range(cfg.max_epochs):
        perm = torch.randperm(n, generator=gen)
        epoch_loss, batches = 0.0, 0
        for s in range(0, n, cfg.batch_size):
            idx = perm[s:s + cfg.batch_size]
            opt.zero_grad()
            logits = net(xt[idx], logits=True)
            loss = (F.cross_entropy(logits, targets[idx]) if use_ce
                    else F.mse_loss(logits, yt[idx]))
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            batches += 1
        avg = epoch_loss / max(batches, 1)
        loss_history.append(avg)

        if tol is not None:
            if last is not None and abs(last - avg) > tol:
                stable = 0
            stable += 1
            last = avg
            if stable == EARLY_STOP_WINSIZE:
                break

    net.eval()
    with torch.no_grad():
        xtt = torch.tensor(np.asarray(test_x, dtype=np.float32))
        ytt = torch.tensor(np.asarray(test_y, dtype=np.float32)).argmax(dim=1)
        accuracy = (net(xtt, logits=True).argmax(dim=1) == ytt).float().mean().item()
    return loss_history, accuracy


# ── Docker lifecycle (own compose, project ono-mono) ──────────────────────────

def node_addrs(n: int) -> list[str]:
    return [f"node-{i}:{NODE_BASE_PORT + i}" for i in range(n)]


def nodes_for(run: dict) -> int:
    return int(run.get("workers", 0)) + int(run.get("servers", 0))


def _compose_dict(nodes: int) -> dict:
    services = {}
    for i in range(nodes):
        port = NODE_BASE_PORT + i
        services[f"node-{i}"] = {
            "container_name": f"node-{i}",
            # Reference the pre-built shared image (see build_node_image); do NOT
            # build per-service or N parallel Rust compiles will OOM the host.
            "image": NODE_IMAGE,
            "ports": [f"{port}:{port}"],
            "networks": ["ono-net"],
            "environment": {
                "HOST": "0.0.0.0",
                "PORT": port,
                "RUST_LOG": "info",
            },
        }
    return {
        "name": COMPOSE_PROJECT,
        "services": services,
        "networks": {"ono-net": {"driver": "bridge"}},
    }


def gen_compose(nodes: int) -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    COMPOSE_FILE.write_text(json.dumps(_compose_dict(nodes), indent=2))


def _compose(*args, **kw):
    return subprocess.run(
        ["docker", "compose", "-p", COMPOSE_PROJECT, "-f", str(COMPOSE_FILE), *args],
        **kw)


def docker_down() -> None:
    if COMPOSE_FILE.exists():
        _compose("down", "--remove-orphans", check=False, capture_output=True)
    time.sleep(1)


def build_node_image(timeout: float = 1800.0) -> None:
    """Build the single shared node image (one Rust compile, reuses BuildKit
    cache). All node-i containers run this same image."""
    env = {**os.environ, "DOCKER_BUILDKIT": "1"}
    subprocess.run(
        ["docker", "build", "-t", NODE_IMAGE,
         "--build-arg", "MODE=release",
         "-f", str(REPO_ROOT / "node" / "Dockerfile"), str(REPO_ROOT)],
        check=True, timeout=timeout, env=env)


def docker_up(nodes: int, build_timeout: float = 1800.0) -> None:
    """Build the shared image (once) and bring the cluster up."""
    build_node_image(timeout=build_timeout)
    gen_compose(nodes)
    _compose("up", "-d", "--remove-orphans", check=True, timeout=300)


def _containers_running(nodes: int) -> bool:
    for i in range(nodes):
        r = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", f"node-{i}"],
                           capture_output=True, text=True)
        if r.returncode != 0 or r.stdout.strip() != "true":
            return False
    return True


def wait_nodes_ready(nodes: int, timeout: float = 90.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _containers_running(nodes):
            time.sleep(2)  # small grace for the listeners to bind
            return True
        time.sleep(1.0)
    return False


def detect_switch(nodes: int) -> bool:
    """True if any node logged the all-reduce -> PS switch (strategy_switch runs)."""
    for i in range(nodes):
        r = subprocess.run(["docker", "logs", f"node-{i}"], capture_output=True, text=True)
        if SWITCH_MARKER in (r.stdout + r.stderr):
            return True
    return False


def capture_logs(nodes: int, label: str) -> None:
    log_dir = DOCKER_LOGS_DIR / label
    log_dir.mkdir(parents=True, exist_ok=True)
    for i in range(nodes):
        r = subprocess.run(["docker", "logs", f"node-{i}"], capture_output=True, text=True)
        (log_dir / f"node-{i}.log").write_text(r.stdout + r.stderr)


# ── Training config builder ───────────────────────────────────────────────────

def _ps_store_sync():
    """This project ALWAYS uses barrier + blocking for PS / strategy-switch."""
    from orchestra.store import BlockingStore
    from orchestra.sync import BarrierSync
    return BlockingStore(), BarrierSync()


def build_training(run: dict, train_s: Path, train_l: Path, x_size: int, y_size: int):
    import orchestra
    from orchestra.datasets import LocalDataset
    from orchestra.loss_fns import CrossEntropy, Mse
    from orchestra.optimizers import GradientDescent

    addrs = node_addrs(nodes_for(run))
    loss_fn = CrossEntropy() if run["loss_fn"] == "cross_entropy" else Mse()
    common = dict(
        addrs=addrs,
        dataset=LocalDataset(str(train_s), str(train_l), x_size=x_size, y_size=y_size),
        optimizer=GradientDescent(lr=run["lr"]),
        loss_fn=loss_fn,
        max_epochs=run["max_epochs"],
        batch_size=run["batch_size"],
        offline_epochs=run.get("offline_epochs", 0),
        seed=run.get("seed", SEED),
        early_stopping_tolerance=run.get("early_stopping_tolerance"),
    )
    strategy = run["strategy"]
    if strategy == "all_reduce":
        return orchestra.all_reduce(**common)
    store, sync = _ps_store_sync()
    extra = dict(nservers=run["servers"], sync=sync, store=store)
    if strategy == "parameter_server":
        return orchestra.parameter_server(**common, **extra)
    if strategy == "strategy_switch":
        return orchestra.strategy_switch(**common, **extra)
    raise ValueError(f"unknown strategy {strategy!r}")


# ── Evaluation ────────────────────────────────────────────────────────────────

def _read_arrays(s_path: Path, l_path: Path, x_size: int, y_size: int):
    import numpy as np
    x = np.fromfile(str(s_path), dtype=np.float32).reshape(-1, x_size)
    y = np.fromfile(str(l_path), dtype=np.float32).reshape(-1, y_size)
    return x, y


def evaluate(model_name, sf_path: Path, test_s: Path, test_l: Path,
             x_size: int, y_size: int) -> float:
    import numpy as np
    import torch
    from safetensors.torch import load_file

    net = build_torch_ref(model_name, load_file(str(sf_path)))
    x = np.fromfile(str(test_s), dtype=np.float32).reshape(-1, x_size)
    y = np.fromfile(str(test_l), dtype=np.float32).reshape(-1, y_size)
    xt = torch.tensor(x)
    yt = torch.tensor(y.argmax(axis=1), dtype=torch.long)
    with torch.no_grad():
        return (net(xt).argmax(dim=1) == yt).float().mean().item()


# ── Derived metrics (replicated) ──────────────────────────────────────────────

def derive(result: dict) -> dict:
    secs = result.get("train_seconds")
    losses = result.get("loss_history") or []
    samples = result.get("train_samples")

    result["epochs_ran"] = len(losses)
    result["final_loss"] = losses[-1] if losses else None
    result["epochs_per_sec"] = (len(losses) / secs) if secs else None

    if samples and secs:
        result["samples_per_sec"] = (samples * len(losses)) / secs
    else:
        result["samples_per_sec"] = None

    if losses and secs:
        result["loss_per_sec"] = (losses[0] - losses[-1]) / secs
    else:
        result["loss_per_sec"] = None

    acc = result.get("test_accuracy")
    result["accuracy_per_sec"] = (acc / secs) if (acc is not None and secs) else None
    return result


# ── Result schema ─────────────────────────────────────────────────────────────

def _synchronizer(run: dict):
    if run["strategy"] in ("parameter_server", "strategy_switch"):
        return "barrier"
    return None


def _store(run: dict):
    if run["strategy"] in ("parameter_server", "strategy_switch"):
        return "blocking"
    return None


def _effective_global_batch(run: dict) -> int:
    bs = int(run["batch_size"])
    if run["strategy"] == "baseline":
        return bs
    return int(run.get("workers", 1)) * bs


def _base_result(run: dict) -> dict:
    return {
        "dataset": run.get("dataset"),
        "model": run.get("model"),
        "strategy": run.get("strategy"),
        "workers": run.get("workers", 0),
        "servers": run.get("servers", 0),
        "synchronizer": _synchronizer(run),
        "store": _store(run),
        "batch_size_per_worker": run.get("batch_size"),
        "effective_global_batch": _effective_global_batch(run),
        "optimizer": "sgd",
        "learning_rate": run.get("lr"),
        "seed": run.get("seed", SEED),
        "max_epochs": run.get("max_epochs"),
        "offline_epochs": run.get("offline_epochs", 0),
        "loss_history": None,
        "epochs_ran": 0,
        "final_loss": None,
        "test_accuracy": None,
        "train_seconds": None,
        "samples_per_sec": None,
        "switched": None,
        "subset": run.get("subset"),
        "label": run.get("label"),
        "notes": run.get("notes"),
        "status": "ok",
        "error": None,
        # internal helpers (kept in the record; harmless extra fields)
        "train_samples": None,
    }


def _run_key(run: dict) -> str:
    return (f"{run.get('label') or 'run'}|{run['dataset']}|{run['model']}|"
            f"{run['strategy']}|{run.get('workers',0)}w{run.get('servers',0)}s|"
            f"b{run['batch_size']}|o{run.get('offline_epochs',0)}|s{run.get('seed',SEED)}")


# ── Single run ────────────────────────────────────────────────────────────────

def run_baseline(run: dict) -> dict:
    """Single-process PyTorch reference (no Docker, no orchestra)."""
    result = _base_result(run)
    cfg = TrainingConfig(lr=run["lr"], batch_size=run["batch_size"],
                         max_epochs=run["max_epochs"], seed=run.get("seed", SEED),
                         early_stopping_tolerance=run.get("early_stopping_tolerance"))
    try:
        train_s, train_l, test_s, test_l, x_size, y_size = resolve_dataset(
            run["dataset"], run.get("subset"))
        result["train_samples"] = _sample_count(train_s, x_size)
        train_x, train_y = _read_arrays(train_s, train_l, x_size, y_size)
        test_x, test_y = _read_arrays(test_s, test_l, x_size, y_size)
        t0 = time.perf_counter()
        loss_history, accuracy = train_torch_baseline(
            run["model"], train_x, train_y, test_x, test_y, cfg)
        result["train_seconds"] = time.perf_counter() - t0
        result["loss_history"] = loss_history
        result["test_accuracy"] = accuracy if run.get("eval") else None
    except Exception:
        result["status"] = "error"
        result["error"] = traceback.format_exc()
    return derive(result)


def run_single(run: dict) -> dict:
    """Run one configuration. For distributed strategies the cluster of
    ``nodes_for(run)`` containers MUST already be up (run_experiment manages
    that). Returns the raw result dict (see schema)."""
    if run["strategy"] == "baseline":
        return run_baseline(run)

    result = _base_result(run)
    nodes = nodes_for(run)
    try:
        train_s, train_l, test_s, test_l, x_size, y_size = resolve_dataset(
            run["dataset"], run.get("subset"))
        result["train_samples"] = _sample_count(train_s, x_size)
        model = build_orchestra_model(run["model"])
        training = build_training(run, train_s, train_l, x_size, y_size)

        import orchestra
        t0 = time.perf_counter()
        trained = orchestra.orchestrate(model, training).wait()
        result["train_seconds"] = time.perf_counter() - t0
        result["loss_history"] = trained.loss_history()

        if run["strategy"] == "strategy_switch":
            result["switched"] = detect_switch(nodes)

        if run.get("eval"):
            ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
            key = _run_key(run).replace("|", "_").replace("/", "-")
            sf = ARTIFACTS_DIR / f"{key}.safetensors"
            trained.save_safetensors(str(sf))
            result["test_accuracy"] = evaluate(
                run["model"], sf, test_s, test_l, x_size, y_size)
    except Exception:
        result["status"] = "error"
        result["error"] = traceback.format_exc()
    return derive(result)
