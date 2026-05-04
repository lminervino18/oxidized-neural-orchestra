#!/usr/bin/env python3
"""
MNIST end-to-end benchmark for O.N.O distributed training.

Usage:
  python benchmarks/mnist_e2e.py --profile smoke
  python benchmarks/mnist_e2e.py --profile benchmark
  python benchmarks/mnist_e2e.py --profile smoke --output benchmarks/results/out.jsonl
  python benchmarks/mnist_e2e.py --profile smoke --keep-containers
  python benchmarks/mnist_e2e.py --profile smoke --rebuild
"""

import argparse
import datetime
import getpass
import gzip
import json
import os
import shutil
import struct
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT     = Path(__file__).resolve().parent.parent
DOCKER_DIR    = REPO_ROOT / "docker"
COMPOSE_FILE  = REPO_ROOT / "compose.yaml"
NB_DATA_DIR   = REPO_ROOT / "notebooks" / "mnist" / "data"
BENCH_DIR     = Path(__file__).resolve().parent
RESULTS_DIR   = BENCH_DIR / "results"
PLOTS_DIR     = BENCH_DIR / "plots"
ARTIFACTS_DIR = RESULTS_DIR / "artifacts"
SUBSET_DIR    = RESULTS_DIR / "data"

X_SIZE = 784
Y_SIZE = 10

MNIST_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images":  "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels":  "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}

# ── Model architecture definitions ────────────────────────────────────────────
#
# dense: first layer gets input_size=784 implicitly from the training dataset.
# conv:  flat_size must match the flattened output of the last conv layer.
#        For conv_tiny: 8 filters, (28-4)//2+1=13 → 8*13*13=1352.
#        kernel=4 required: (28-4)%2==0 → backward pass doesn't panic.

MODEL_DEFS = {
    "dense_tiny": {
        "type": "dense",
        "dense": [
            {"size": 64,  "act": "sigmoid"},
            {"size": 10,  "act": "sigmoid"},
        ],
    },
    "dense_small": {
        "type": "dense",
        "dense": [
            {"size": 128, "act": "sigmoid"},
            {"size": 64,  "act": "sigmoid"},
            {"size": 10,  "act": "sigmoid"},
        ],
    },
    "conv_tiny": {
        "type": "conv",
        "conv": [
            {
                "input_dim": (1, 28, 28),
                "kernel_dim": (8, 1, 4),
                "stride": 2,
                "padding": 0,
                "act": None,
            },
        ],
        "dense": [{"size": 10, "act": "sigmoid"}],
        "flat_size": 8 * 13 * 13,
    },
}

# ── Run generation ────────────────────────────────────────────────────────────


def build_runs(profile_cfg: dict, all_presets: dict) -> list:
    """
    Generate the run list as a cartesian product of models × training_presets.

    Override priority (highest wins): combo (model+preset) > model > preset > defaults.
    """
    defaults  = profile_cfg["defaults"]
    overrides = profile_cfg.get("overrides", {})

    runs = []
    for preset_name in profile_cfg["training_presets"]:
        preset = all_presets[preset_name]
        for model in profile_cfg["models"]:
            cfg = dict(defaults)
            cfg.update(overrides.get(preset_name, {}))
            cfg.update(overrides.get(model, {}))
            cfg.update(overrides.get(f"{model}+{preset_name}", {}))

            cfg["name"]       = f"{model}_{preset_name}"
            cfg["model"]      = model
            cfg["training"]   = preset_name
            cfg["workers"]    = preset["workers"]
            cfg["servers"]    = preset["servers"]
            cfg["serializer"] = preset["serializer"]
            cfg["sync"]       = preset["sync"]
            cfg["store"]      = preset["store"]
            runs.append(cfg)

    return runs


# ── Orchestra import ──────────────────────────────────────────────────────────


def _ensure_orchestra():
    try:
        import orchestra  # noqa: F401
        return
    except ImportError:
        pass
    # Try repo-root .venv first (has torch + safetensors + orchestra)
    for venv in [REPO_ROOT / ".venv", REPO_ROOT / "orchestra-py" / ".venv"]:
        for site in (venv / "lib").glob("python*/site-packages"):
            sys.path.insert(0, str(site))
    try:
        import orchestra  # noqa: F401
    except ImportError:
        sys.exit(
            "ERROR: orchestra is not importable.\n"
            "Run with: .venv/bin/python benchmarks/mnist_e2e.py ..."
        )


# ── MNIST dataset preparation ─────────────────────────────────────────────────


def _dl_raw(name: str, url: str, raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / name
    if dest.exists():
        return dest
    print(f"  downloading {name}...")
    gz = raw_dir / (name + ".gz")
    urllib.request.urlretrieve(url, gz)
    with gzip.open(gz, "rb") as fi, open(dest, "wb") as fo:
        shutil.copyfileobj(fi, fo)
    gz.unlink()
    return dest


def _read_images(path: Path):
    with open(path, "rb") as f:
        _, n, rows, cols = struct.unpack(">IIII", f.read(16))
        raw = f.read(n * rows * cols)
    ppi = rows * cols
    return [[b / 255.0 for b in raw[i * ppi: (i + 1) * ppi]] for i in range(n)]


def _read_labels(path: Path):
    with open(path, "rb") as f:
        _, n = struct.unpack(">II", f.read(8))
        return list(f.read(n))


def _write_bins(images, labels, s_path: Path, l_path: Path):
    s_path.parent.mkdir(parents=True, exist_ok=True)
    l_path.parent.mkdir(parents=True, exist_ok=True)
    with open(s_path, "wb") as fs, open(l_path, "wb") as fl:
        for img, lbl in zip(images, labels):
            one_hot = [0.0] * Y_SIZE
            one_hot[lbl] = 1.0
            fs.write(struct.pack(f"{X_SIZE}f", *img))
            fl.write(struct.pack(f"{Y_SIZE}f", *one_hot))


def _copy_subset(src_s: Path, src_l: Path, dst_s: Path, dst_l: Path, n: int):
    SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    with open(src_s, "rb") as f:
        data = f.read(n * X_SIZE * 4)
    with open(dst_s, "wb") as f:
        f.write(data)
    with open(src_l, "rb") as f:
        data = f.read(n * Y_SIZE * 4)
    with open(dst_l, "wb") as f:
        f.write(data)


def prepare_dataset(train_n, test_n):
    """
    Ensure dataset binary files exist.
    Returns (train_samples, train_labels, test_samples, test_labels) as absolute Paths.

    Reuses notebooks/mnist/data/ if files already exist (notebooks download them).
    Creates subsets in benchmarks/results/data/ for smoke-sized runs.
    """
    raw_dir = NB_DATA_DIR / "mnist_raw"
    full = {
        "train_s": NB_DATA_DIR / "mnist_train_samples.bin",
        "train_l": NB_DATA_DIR / "mnist_train_labels.bin",
        "test_s":  NB_DATA_DIR / "mnist_test_samples.bin",
        "test_l":  NB_DATA_DIR / "mnist_test_labels.bin",
    }

    if not (full["train_s"].exists() and full["train_l"].exists()):
        print("Preparing full MNIST training set (runs once)...")
        _dl_raw("train_images", MNIST_URLS["train_images"], raw_dir)
        _dl_raw("train_labels", MNIST_URLS["train_labels"], raw_dir)
        _write_bins(
            _read_images(raw_dir / "train_images"),
            _read_labels(raw_dir / "train_labels"),
            full["train_s"], full["train_l"],
        )

    if not (full["test_s"].exists() and full["test_l"].exists()):
        print("Preparing full MNIST test set (runs once)...")
        _dl_raw("test_images", MNIST_URLS["test_images"], raw_dir)
        _dl_raw("test_labels", MNIST_URLS["test_labels"], raw_dir)
        _write_bins(
            _read_images(raw_dir / "test_images"),
            _read_labels(raw_dir / "test_labels"),
            full["test_s"], full["test_l"],
        )

    full_train_n = full["train_s"].stat().st_size // (X_SIZE * 4)
    full_test_n  = full["test_s"].stat().st_size // (X_SIZE * 4)

    train_s, train_l = full["train_s"], full["train_l"]
    test_s,  test_l  = full["test_s"],  full["test_l"]

    if train_n and train_n < full_train_n:
        sub_s = SUBSET_DIR / f"mnist_train_{train_n}_samples.bin"
        sub_l = SUBSET_DIR / f"mnist_train_{train_n}_labels.bin"
        if not (sub_s.exists() and sub_l.exists()):
            print(f"  creating training subset ({train_n} samples)...")
            _copy_subset(full["train_s"], full["train_l"], sub_s, sub_l, train_n)
        train_s, train_l = sub_s, sub_l

    if test_n and test_n < full_test_n:
        sub_s = SUBSET_DIR / f"mnist_test_{test_n}_samples.bin"
        sub_l = SUBSET_DIR / f"mnist_test_{test_n}_labels.bin"
        if not (sub_s.exists() and sub_l.exists()):
            print(f"  creating test subset ({test_n} samples)...")
            _copy_subset(full["test_s"], full["test_l"], sub_s, sub_l, test_n)
        test_s, test_l = sub_s, sub_l

    return train_s, train_l, test_s, test_l


# ── Docker management ─────────────────────────────────────────────────────────


def _hosts_ok(workers: int, servers: int) -> bool:
    try:
        content = Path("/etc/hosts").read_text()
        return (
            all(f"worker-{i}" in content for i in range(workers))
            and all(f"server-{i}" in content for i in range(servers))
        )
    except OSError:
        return False


def ensure_hosts(workers: int, servers: int, sudo_pw: str):
    if _hosts_ok(workers, servers):
        return
    print(f"  updating /etc/hosts for {workers}w/{servers}s topology...")
    env = {**os.environ, "WORKERS": str(workers), "SERVERS": str(servers)}
    result = subprocess.run(
        ["sudo", "-S", "-E", "python3", str(DOCKER_DIR / "fill_hosts.py")],
        input=sudo_pw + "\n",
        text=True,
        env=env,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"fill_hosts failed:\n{result.stderr}")


def _gen_compose(workers: int, servers: int, release: bool):
    env = {
        **os.environ,
        "WORKERS": str(workers),
        "SERVERS": str(servers),
        "RELEASE": str(release).lower(),
    }
    subprocess.run(["python3", str(DOCKER_DIR / "gen_compose.py")], env=env, check=True)


def docker_up(workers: int, servers: int, rebuild: bool, release: bool = True) -> float:
    """Start containers. Returns elapsed seconds (docker startup time)."""
    t0 = time.perf_counter()
    _gen_compose(workers, servers, release)
    cmd = ["docker", "compose", "-f", str(COMPOSE_FILE), "up"]
    cmd += ["--build", "--no-cache"] if rebuild else ["--build"]
    cmd += ["-d", "--remove-orphans"]
    subprocess.run(cmd, check=True)
    time.sleep(3)  # build step (COPY context) already provides ~15-20s; 3s covers node startup
    return time.perf_counter() - t0



def docker_down():
    subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), "down"],
        check=True,
        capture_output=True,
    )


def containers_running(workers: int, servers: int) -> bool:
    names = [f"server-{i}" for i in range(servers)] + [f"worker-{i}" for i in range(workers)]
    for name in names:
        r = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", name],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0 or r.stdout.strip() != "true":
            return False
    return True


def wait_nodes_ready(workers: int, servers: int, timeout: float = 30.0) -> bool:
    """Wait until all node containers are Running, up to ``timeout`` seconds."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if containers_running(workers, servers):
            return True
        time.sleep(1.0)
    return False


# ── Model building ────────────────────────────────────────────────────────────


def build_model(model_name: str):
    from orchestra import Sequential
    from orchestra.arch import Conv2d, Dense
    from orchestra.activations import Sigmoid
    from orchestra.initialization import Xavier

    def act(name):
        return Sigmoid() if name == "sigmoid" else None

    mdef = MODEL_DEFS[model_name]
    if mdef["type"] == "dense":
        layers = [Dense(l["size"], Xavier(), act(l["act"])) for l in mdef["dense"]]
    else:
        layers = [
            Conv2d(
                input_dim=l["input_dim"],
                kernel_dim=l["kernel_dim"],
                stride=l["stride"],
                padding=l["padding"],
                init=Xavier(),
                act_fn=act(l["act"]),
            )
            for l in mdef["conv"]
        ] + [Dense(l["size"], Xavier(), act(l["act"])) for l in mdef["dense"]]

    return Sequential(layers)


def build_training(run_cfg: dict, train_s: Path, train_l: Path):
    import orchestra
    from orchestra.datasets import LocalDataset
    from orchestra.loss_fns import Mse
    from orchestra.optimizers import GradientDescent
    from orchestra.serializer import BaseSerializer, SparseSerializer
    from orchestra.store import BlockingStore, WildStore
    from orchestra.sync import BarrierSync, NonBlockingSync

    w, s = run_cfg["workers"], run_cfg["servers"]
    worker_addrs = [f"worker-{i}:{50000 + i}" for i in range(w)]
    server_addrs = [f"server-{i}:{40000 + i}" for i in range(s)]

    sync  = BarrierSync()    if run_cfg["sync"]       == "barrier"  else NonBlockingSync()
    store = BlockingStore()  if run_cfg["store"]      == "blocking" else WildStore()
    ser   = BaseSerializer() if run_cfg["serializer"] == "base"     else SparseSerializer(r=0.9)

    return orchestra.parameter_server(
        worker_addrs=worker_addrs,
        server_addrs=server_addrs,
        dataset=LocalDataset(str(train_s), str(train_l), x_size=X_SIZE, y_size=Y_SIZE),
        optimizer=GradientDescent(lr=run_cfg.get("lr", 0.5)),
        loss_fn=Mse(),
        sync=sync,
        store=store,
        serializer=ser,
        max_epochs=run_cfg["max_epochs"],
        batch_size=run_cfg["batch_size"],
        offline_epochs=run_cfg.get("offline_epochs", 0),
        seed=42,
        early_stopping_tolerance=run_cfg.get("early_stopping_tolerance"),
    )


# ── Accuracy evaluation ───────────────────────────────────────────────────────


def evaluate(model_name: str, safetensors_path: Path, test_s: Path, test_l: Path) -> float:
    """
    Load trained weights into an equivalent PyTorch model and compute top-1 accuracy.

    Weight layout from ONO safetensors:
      Dense:  layer_N.weight = [in, out]  → PyTorch Linear expects [out, in], needs .T
      Conv2d: layer_N.weight = [filters, in_ch, kH, kW] → same as PyTorch, no transpose
    """
    import numpy as np
    import torch
    import torch.nn as nn
    from safetensors.torch import load_file

    mdef = MODEL_DEFS[model_name]
    sd   = load_file(str(safetensors_path))

    def _act(name):
        return torch.sigmoid if name == "sigmoid" else None

    if mdef["type"] == "dense":
        dense_defs = mdef["dense"]
        sizes = [X_SIZE] + [l["size"] for l in dense_defs]

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([
                    nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)
                ])

            def forward(self, x):
                for i, layer in enumerate(self.layers):
                    x = layer(x)
                    fn = _act(dense_defs[i]["act"])
                    if fn is not None:
                        x = fn(x)
                return x

        net = Net()
        with torch.no_grad():
            for j, layer in enumerate(net.layers):
                layer.weight.copy_(sd[f"layer_{j}.weight"].T)
                layer.bias.copy_(sd[f"layer_{j}.bias"])

    else:
        conv_defs   = mdef["conv"]
        dense_defs  = mdef["dense"]
        flat_size   = mdef["flat_size"]
        dense_sizes = [flat_size] + [l["size"] for l in dense_defs]

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.convs = nn.ModuleList([
                    nn.Conv2d(
                        in_channels=l["kernel_dim"][1],
                        out_channels=l["kernel_dim"][0],
                        kernel_size=l["kernel_dim"][2],
                        stride=l["stride"],
                        padding=l["padding"],
                    )
                    for l in conv_defs
                ])
                self.linears = nn.ModuleList([
                    nn.Linear(dense_sizes[i], dense_sizes[i + 1])
                    for i in range(len(dense_defs))
                ])

            def forward(self, x):
                x = x.view(-1, 1, 28, 28)
                for i, conv in enumerate(self.convs):
                    x = conv(x)
                    fn = _act(conv_defs[i]["act"])
                    if fn is not None:
                        x = fn(x)
                x = x.flatten(start_dim=1)
                for i, linear in enumerate(self.linears):
                    x = linear(x)
                    fn = _act(dense_defs[i]["act"])
                    if fn is not None:
                        x = fn(x)
                return x

        net = Net()
        with torch.no_grad():
            for i, conv in enumerate(net.convs):
                conv.weight.copy_(sd[f"layer_{i}.weight"])
                conv.bias.copy_(sd[f"layer_{i}.bias"])
            offset = len(conv_defs)
            for j, linear in enumerate(net.linears):
                linear.weight.copy_(sd[f"layer_{offset + j}.weight"].T)
                linear.bias.copy_(sd[f"layer_{offset + j}.bias"])

    net.eval()
    x_raw = np.fromfile(str(test_s), dtype=np.float32).reshape(-1, X_SIZE)
    y_raw = np.fromfile(str(test_l), dtype=np.float32).reshape(-1, Y_SIZE)
    x_t   = torch.tensor(x_raw)
    y_t   = torch.tensor(y_raw.argmax(axis=1), dtype=torch.long)

    with torch.no_grad():
        return (net(x_t).argmax(dim=1) == y_t).float().mean().item()


# ── Single benchmark run ──────────────────────────────────────────────────────


def run_single(run_cfg: dict, profile_cfg: dict, profile_name: str) -> dict:
    """Execute one benchmark run. docker_start_seconds is filled by the caller."""
    model_name    = run_cfg["model"]
    training_name = run_cfg["training"]
    train_n       = profile_cfg.get("train_samples")
    test_n        = profile_cfg.get("test_samples")

    run_id = (
        datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        + f"_{model_name}_{training_name}"
    )

    result = {
        "run_id":               run_id,
        "profile":              profile_name,
        "model_name":           model_name,
        "training_name":        training_name,
        "workers":              run_cfg["workers"],
        "servers":              run_cfg["servers"],
        "serializer":           run_cfg["serializer"],
        "sync":                 run_cfg["sync"],
        "store":                run_cfg["store"],
        "max_epochs":           run_cfg["max_epochs"],
        "batch_size":           run_cfg["batch_size"],
        "lr":                   run_cfg.get("lr", 0.5),
        "offline_epochs":       run_cfg.get("offline_epochs", 0),
        "train_samples":        train_n,
        "test_samples":         test_n,
        "docker_start_seconds": None,
        "train_seconds":        None,
        "eval_seconds":         None,
        "accuracy":             None,
        "min_accuracy":         run_cfg["min_accuracy"],
        "max_train_seconds":    run_cfg["max_train_seconds"],
        "passed":               False,
        "error":                None,
    }

    try:
        train_s, train_l, test_s, test_l = prepare_dataset(train_n, test_n)

        model    = build_model(model_name)
        training = build_training(run_cfg, train_s, train_l)

        # ── Training (measured) ───────────────────────────────────────────────
        import orchestra
        t0 = time.perf_counter()
        session = orchestra.orchestrate(model, training)
        trained = session.wait()
        result["train_seconds"] = time.perf_counter() - t0

        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        sf_path = ARTIFACTS_DIR / f"{run_id}.safetensors"
        trained.save_safetensors(str(sf_path))

        # ── Evaluation (measured separately) ─────────────────────────────────
        t_eval = time.perf_counter()
        result["accuracy"] = evaluate(model_name, sf_path, test_s, test_l)
        result["eval_seconds"] = time.perf_counter() - t_eval

        result["passed"] = (
            result["accuracy"] >= run_cfg["min_accuracy"]
            and result["train_seconds"] <= run_cfg["max_train_seconds"]
        )

    except Exception:
        result["error"] = traceback.format_exc()

    return result


# ── Results I/O ───────────────────────────────────────────────────────────────


def save_result(result: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(result) + "\n")


def print_summary(results: list):
    W = 72
    print("\n" + "─" * W)
    print(f"{'MODEL':<14} {'TRAINING':<30} {'TIME(s)':>8} {'ACC':>6} {'PASS':>5}")
    print("─" * W)
    for r in results:
        t = f"{r['train_seconds']:.2f}" if r["train_seconds"] is not None else "  ERROR"
        a = f"{r['accuracy']:.3f}"      if r["accuracy"]       is not None else "   N/A"
        p = "yes" if r["passed"] else "no"
        print(f"{r['model_name']:<14} {r['training_name']:<30} {t:>8} {a:>6} {p:>5}")
    print("─" * W)
    ok = sum(r["passed"] for r in results)
    print(f"\n{ok}/{len(results)} passed.\n")


# ── Plots ────────────────────────────────────────────────────────────────────


def plot_results(results: list, profile: str, run_ts: str):
    """
    Generate a two-panel summary figure (accuracy + training time) and save it to
    results/latest/{profile}_results.png, overwriting any previous run.

    The README references this fixed path, so it always shows the latest results.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available — skipping plots)")
        return

    completed = [r for r in results if r.get("train_seconds") is not None]
    if not completed:
        return

    labels      = [f"{r['model_name']}\n{r['workers']}w/{r['servers']}s" for r in completed]
    accuracies  = [r["accuracy"] if r["accuracy"] is not None else 0.0 for r in completed]
    train_times = [r["train_seconds"] for r in completed]
    colors      = ["#4caf50" if r["passed"] else "#ef5350" for r in completed]

    # Use the lowest min_accuracy threshold as the single reference line
    min_acc_thresh = min(
        r.get("min_accuracy", 0.0) for r in completed if r.get("min_accuracy") is not None
    )

    fig, (ax_acc, ax_time) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"O.N.O MNIST Benchmark — profile={profile}  ({run_ts})",
        fontsize=13, fontweight="bold", y=1.02,
    )

    x     = list(range(len(completed)))
    bar_w = 0.55

    # ── Accuracy subplot ──────────────────────────────────────────────────────
    bars = ax_acc.bar(x, accuracies, width=bar_w, color=colors, edgecolor="white", linewidth=0.8)
    ax_acc.axhline(
        min_acc_thresh, color="#e53935", linestyle="--", linewidth=1.3,
        label=f"min_accuracy = {min_acc_thresh:.2f}",
    )
    for bar, val in zip(bars, accuracies):
        if val > 0:
            ax_acc.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=9,
            )
    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels(labels, fontsize=9)
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0, 1.0)
    ax_acc.set_title("Test Accuracy")
    ax_acc.legend(fontsize=8)
    ax_acc.grid(axis="y", alpha=0.3)
    ax_acc.axhline(0, color="black", linewidth=0.6)

    # ── Training time subplot ─────────────────────────────────────────────────
    bars = ax_time.bar(x, train_times, width=bar_w, color=colors, edgecolor="white", linewidth=0.8)
    offset = max(train_times) * 0.025 if train_times else 0.05
    for bar, val in zip(bars, train_times):
        ax_time.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{val:.2f}s",
            ha="center", va="bottom", fontsize=9,
        )
    ax_time.set_xticks(x)
    ax_time.set_xticklabels(labels, fontsize=9)
    ax_time.set_ylabel("Seconds")
    ax_time.set_ylim(0, max(max(train_times) * 1.35, 0.5))
    ax_time.set_title("Training Time (Docker startup excluded)")
    ax_time.grid(axis="y", alpha=0.3)
    ax_time.axhline(0, color="black", linewidth=0.6)

    # Shared legend for pass/fail
    fig.legend(
        handles=[
            mpatches.Patch(color="#4caf50", label="pass"),
            mpatches.Patch(color="#ef5350", label="fail"),
        ],
        loc="lower center", ncol=2,
        bbox_to_anchor=(0.5, -0.06), fontsize=9,
    )

    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / f"{profile}_results.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  plots → {out.relative_to(BENCH_DIR)}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="MNIST end-to-end benchmark for O.N.O")
    p.add_argument("--profile",         choices=["smoke", "benchmark"], default="smoke")
    p.add_argument("--config",          type=Path,
                   default=Path(__file__).parent / "mnist_configs.json")
    p.add_argument("--output",          type=Path)
    p.add_argument("--keep-containers", action="store_true",
                   help="Leave Docker containers running after benchmark")
    p.add_argument("--rebuild",         action="store_true",
                   help="Force Docker image rebuild (passes --no-cache)")
    return p.parse_args()


def main():
    args = parse_args()
    _ensure_orchestra()

    with open(args.config) as f:
        config = json.load(f)

    profile_cfg  = config["profiles"][args.profile]
    all_presets  = config["training_presets"]
    runs         = build_runs(profile_cfg, all_presets)

    if args.output is None:
        ts = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        args.output = RESULTS_DIR / f"{ts}_{args.profile}.jsonl"

    # Group runs by topology so we can reuse Docker containers
    by_topo: dict = {}
    for run in runs:
        key = (run["workers"], run["servers"])
        by_topo.setdefault(key, []).append(run)

    # Determine if /etc/hosts needs updating (ask for sudo once, upfront)
    max_w = max(k[0] for k in by_topo)
    max_s = max(k[1] for k in by_topo)
    sudo_pw = ""
    if not _hosts_ok(max_w, max_s):
        sudo_pw = getpass.getpass(
            f"sudo password (needed to update /etc/hosts for {max_w}w/{max_s}s): "
        )

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
    except Exception:
        commit = "unknown"

    print(f"\nO.N.O MNIST Benchmark — profile={args.profile} — commit={commit}")
    print(f"runs={len(runs)}  topologies={len(by_topo)}  output={args.output}\n")

    all_results = []

    for (workers, servers), topo_runs in sorted(by_topo.items()):
        topology_id = f"{workers}w_{servers}s"
        print(f"\n── {workers}w / {servers}s " + "─" * 50)

        docker_start_s = 0.0
        running = containers_running(workers, servers)

        try:
            ensure_hosts(workers, servers, sudo_pw)
            if running:
                print("  containers already running, reusing.")
            else:
                print("  starting Docker containers...")
                docker_start_s = docker_up(workers, servers, rebuild=args.rebuild)
                print(f"  Docker ready in {docker_start_s:.1f}s")

        except Exception as e:
            print(f"  ERROR: Docker setup failed: {e}")
            for run in topo_runs:
                result = {
                    "run_id":                 run["name"],
                    "profile":                args.profile,
                    "model_name":             run["model"],
                    "training_name":          run["training"],
                    "workers":                workers,
                    "servers":                servers,
                    "topology_id":            topology_id,
                    "reused_topology":        False,
                    "docker_restarted_for_run": False,
                    "node_reuse_enabled":     True,
                    "docker_start_seconds":   None,
                    "train_seconds":          None,
                    "eval_seconds":           None,
                    "accuracy":               None,
                    "passed":                 False,
                    "error":                  str(e),
                }
                all_results.append(result)
                save_result(result, args.output)
            continue

        save_result({
            "event":                "topology_started",
            "topology_id":         topology_id,
            "docker_start_seconds": docker_start_s,
        }, args.output)

        try:
            for i, run in enumerate(topo_runs):
                docker_restarted = False
                if i > 0 and not wait_nodes_ready(workers, servers, timeout=30.0):
                    print("  WARNING: nodes not ready after previous session — restarting Docker (fallback)")
                    try:
                        docker_down()
                    except Exception:
                        pass
                    docker_start_s = docker_up(workers, servers, rebuild=False)
                    ensure_hosts(workers, servers, sudo_pw)
                    docker_restarted = True
                    print(f"  Docker restarted in {docker_start_s:.1f}s (fallback recovery)")

                print(f"\n  [{run['name']}]")
                result = run_single(run, profile_cfg, args.profile)
                result["docker_start_seconds"]    = docker_start_s if (i == 0 or docker_restarted) else 0.0
                result["topology_id"]             = topology_id
                result["reused_topology"]         = i > 0 and not docker_restarted
                result["docker_restarted_for_run"] = docker_restarted
                result["node_reuse_enabled"]      = True
                all_results.append(result)
                save_result(result, args.output)

                status = "PASS" if result["passed"] else "FAIL"
                ts_str = f"{result['train_seconds']:.2f}s" if result["train_seconds"] else "ERROR"
                ac_str = f"{result['accuracy']:.3f}"       if result["accuracy"] is not None else "N/A"
                reuse_tag   = " [reused]"           if result["reused_topology"]          else ""
                restart_tag = " [fallback-restart]" if result["docker_restarted_for_run"] else ""
                print(f"  → {status}  train={ts_str}  acc={ac_str}{reuse_tag}{restart_tag}")

                # In smoke mode, abort the topology group on first errored run
                if args.profile == "smoke" and not result["passed"] and result["error"]:
                    print("  smoke run errored — stopping early.")
                    break

        finally:
            if not args.keep_containers:
                print("\n  stopping containers...")
                try:
                    docker_down()
                except Exception as ex:
                    print(f"  WARNING: docker down failed: {ex}")

    print_summary(all_results)

    run_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    plot_results(all_results, args.profile, run_ts)

    smoke_failed = args.profile == "smoke" and any(not r["passed"] for r in all_results)
    sys.exit(1 if smoke_failed else 0)


if __name__ == "__main__":
    main()
