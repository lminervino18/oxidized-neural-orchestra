#!/usr/bin/env python3
"""
MNIST end-to-end benchmark for O.N.O distributed training.

Usage:
  python benchmarks/mnist_e2e.py               # run all 14 combinations
  python benchmarks/mnist_e2e.py --runs 1 11   # run specific numbered runs
  python benchmarks/mnist_e2e.py --keep-containers
  python benchmarks/mnist_e2e.py --rebuild
"""

import argparse
import datetime
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

REPO_ROOT       = Path(__file__).resolve().parent.parent
DOCKER_DIR      = REPO_ROOT / "docker"
COMPOSE_FILE    = REPO_ROOT / "compose.yaml"
NB_DATA_DIR     = REPO_ROOT / "notebooks" / "mnist" / "data"
BENCH_DIR       = Path(__file__).resolve().parent
RESULTS_DIR     = BENCH_DIR / "results"
PLOTS_DIR       = BENCH_DIR / "plots"
ARTIFACTS_DIR   = RESULTS_DIR / "artifacts"
SUBSET_DIR      = RESULTS_DIR / "data"
DOCKER_LOGS_DIR = RESULTS_DIR / "docker_logs"

X_SIZE = 784
Y_SIZE = 10

MNIST_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images":  "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels":  "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}

# ── Orchestra import ──────────────────────────────────────────────────────────


def _ensure_orchestra():
    try:
        import orchestra  # noqa: F401
        return
    except ImportError:
        pass
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


# ── /etc/hosts check ──────────────────────────────────────────────────────────


def _hosts_ok(workers: int, servers: int) -> bool:
    try:
        content = Path("/etc/hosts").read_text()
        return (
            all(f"worker-{i}" in content for i in range(workers))
            and all(f"server-{i}" in content for i in range(servers))
        )
    except OSError:
        return False


def check_hosts(runs: list):
    """Fail fast if /etc/hosts is missing required entries for the given runs."""
    max_w = max(r["workers"] for r in runs)
    max_s = max(r.get("servers", 0) for r in runs)
    if _hosts_ok(max_w, max_s):
        return

    content = Path("/etc/hosts").read_text()
    missing = []
    for i in range(max_w):
        if f"worker-{i}" not in content:
            missing.append(f"127.0.0.1 worker-{i}")
    for i in range(max_s):
        if f"server-{i}" not in content:
            missing.append(f"127.0.0.1 server-{i}")

    if missing:
        entries = "\\n".join(missing)
        sys.exit(
            "ERROR: /etc/hosts is missing required entries:\n"
            + "\n".join(f"  {e}" for e in missing)
            + f'\n\nFix with:\n  sudo bash -c "echo \'{chr(10).join(missing)}\' >> /etc/hosts"'
        )


def ensure_hosts_for_topo(workers: int, servers: int):
    if not _hosts_ok(workers, servers):
        raise RuntimeError(
            f"Missing /etc/hosts entries for {workers}w/{servers}s — run check_hosts() first."
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


def _copy_subset(src_s, src_l, dst_s, dst_l, n):
    SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    with open(src_s, "rb") as f:
        data = f.read(n * X_SIZE * 4)
    with open(dst_s, "wb") as f:
        f.write(data)
    with open(src_l, "rb") as f:
        data = f.read(n * Y_SIZE * 4)
    with open(dst_l, "wb") as f:
        f.write(data)


def prepare_dataset(train_n=None, test_n=None):
    raw_dir = NB_DATA_DIR / "mnist_raw"
    full = {
        "train_s": NB_DATA_DIR / "mnist_train_samples.bin",
        "train_l": NB_DATA_DIR / "mnist_train_labels.bin",
        "test_s":  NB_DATA_DIR / "mnist_test_samples.bin",
        "test_l":  NB_DATA_DIR / "mnist_test_labels.bin",
    }
    if not (full["train_s"].exists() and full["train_l"].exists()):
        print("Preparing MNIST training set...")
        _dl_raw("train_images", MNIST_URLS["train_images"], raw_dir)
        _dl_raw("train_labels", MNIST_URLS["train_labels"], raw_dir)
        _write_bins(_read_images(raw_dir/"train_images"), _read_labels(raw_dir/"train_labels"),
                    full["train_s"], full["train_l"])
    if not (full["test_s"].exists() and full["test_l"].exists()):
        print("Preparing MNIST test set...")
        _dl_raw("test_images", MNIST_URLS["test_images"], raw_dir)
        _dl_raw("test_labels", MNIST_URLS["test_labels"], raw_dir)
        _write_bins(_read_images(raw_dir/"test_images"), _read_labels(raw_dir/"test_labels"),
                    full["test_s"], full["test_l"])

    train_s, train_l = full["train_s"], full["train_l"]
    test_s,  test_l  = full["test_s"],  full["test_l"]

    if train_n:
        full_n = full["train_s"].stat().st_size // (X_SIZE * 4)
        if train_n < full_n:
            sub_s = SUBSET_DIR / f"mnist_train_{train_n}_samples.bin"
            sub_l = SUBSET_DIR / f"mnist_train_{train_n}_labels.bin"
            if not (sub_s.exists() and sub_l.exists()):
                _copy_subset(full["train_s"], full["train_l"], sub_s, sub_l, train_n)
            train_s, train_l = sub_s, sub_l

    if test_n:
        full_n = full["test_s"].stat().st_size // (Y_SIZE * 4)
        if test_n < full_n:
            sub_s = SUBSET_DIR / f"mnist_test_{test_n}_samples.bin"
            sub_l = SUBSET_DIR / f"mnist_test_{test_n}_labels.bin"
            if not (sub_s.exists() and sub_l.exists()):
                _copy_subset(full["test_s"], full["test_l"], sub_s, sub_l, test_n)
            test_s, test_l = sub_s, sub_l

    return train_s, train_l, test_s, test_l


# ── Docker management ─────────────────────────────────────────────────────────


def _gen_compose(workers: int, servers: int, release: bool = True):
    env = {**os.environ, "WORKERS": str(workers), "SERVERS": str(servers),
           "RELEASE": str(release).lower()}
    subprocess.run(["python3", str(DOCKER_DIR / "gen_compose.py")], env=env, check=True)


def docker_up(workers: int, servers: int, rebuild: bool, release: bool = True) -> float:
    t0 = time.perf_counter()
    _gen_compose(workers, servers, release)
    if rebuild:
        subprocess.run(
            ["docker", "compose", "-f", str(COMPOSE_FILE), "build", "--no-cache"],
            check=True,
        )
    cmd = ["docker", "compose", "-f", str(COMPOSE_FILE), "up", "--build", "-d", "--remove-orphans"]
    subprocess.run(cmd, check=True)
    time.sleep(10)
    return time.perf_counter() - t0


def docker_down():
    subprocess.run(["docker", "compose", "-f", str(COMPOSE_FILE), "down"],
                   check=True, capture_output=True)
    time.sleep(2)


def containers_running(workers: int, servers: int) -> bool:
    names = [f"server-{i}" for i in range(servers)] + [f"worker-{i}" for i in range(workers)]
    for name in names:
        r = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", name],
                           capture_output=True, text=True)
        if r.returncode != 0 or r.stdout.strip() != "true":
            return False
    return True


def wait_nodes_ready(workers: int, servers: int, timeout: float = 60.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if containers_running(workers, servers):
            return True
        time.sleep(1.0)
    return False


def capture_docker_logs(workers: int, servers: int, label: str):
    DOCKER_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = DOCKER_LOGS_DIR / label
    run_dir.mkdir(exist_ok=True)
    names = [f"worker-{i}" for i in range(workers)] + [f"server-{i}" for i in range(servers)]
    for name in names:
        r = subprocess.run(["docker", "logs", name], capture_output=True, text=True)
        (run_dir / f"{name}.log").write_text(r.stdout + r.stderr)


# ── Model building ────────────────────────────────────────────────────────────


def build_model(model_name: str, model_defs: dict):
    from orchestra import Sequential
    from orchestra.arch import Conv2d, Dense
    from orchestra.activations import Sigmoid, Softmax
    from orchestra.initialization import Xavier

    def act(name):
        if name == "sigmoid":
            return Sigmoid()
        if name == "softmax":
            return Softmax()
        return None

    mdef = model_defs[model_name]
    if mdef["type"] == "dense":
        layers = [Dense(l["size"], Xavier(), act(l["act"])) for l in mdef["dense"]]
    else:
        layers = [
            Conv2d(input_dim=tuple(l["input_dim"]), kernel_dim=tuple(l["kernel_dim"]),
                   stride=l["stride"], padding=l["padding"], init=Xavier(), act_fn=act(l["act"]))
            for l in mdef["conv"]
        ] + [Dense(l["size"], Xavier(), act(l["act"])) for l in mdef["dense"]]
    return Sequential(layers)


def build_training(run_cfg: dict, train_s: Path, train_l: Path):
    import orchestra
    from orchestra.datasets import LocalDataset
    from orchestra.loss_fns import CrossEntropy, Mse
    from orchestra.optimizers import GradientDescent
    from orchestra.serializer import BaseSerializer, SparseSerializer
    from orchestra.store import BlockingStore, WildStore
    from orchestra.sync import BarrierSync, NonBlockingSync

    w, s = run_cfg["workers"], run_cfg.get("servers", 0)
    worker_addrs = [f"worker-{i}:{50000 + i}" for i in range(w)]
    server_addrs = [f"server-{i}:{40000 + i}" for i in range(s)]
    ser     = BaseSerializer() if run_cfg["serializer"] == "base" else SparseSerializer(r=0.9)
    algo    = run_cfg.get("algorithm", "parameter_server")
    loss_fn = CrossEntropy() if run_cfg.get("loss_fn") == "cross_entropy" else Mse()

    common = dict(
        worker_addrs=worker_addrs,
        dataset=LocalDataset(str(train_s), str(train_l), x_size=X_SIZE, y_size=Y_SIZE),
        optimizer=GradientDescent(lr=run_cfg.get("lr", 0.5)),
        loss_fn=loss_fn,
        serializer=ser,
        max_epochs=run_cfg["max_epochs"],
        batch_size=run_cfg["batch_size"],
        offline_epochs=run_cfg.get("offline_epochs", 0),
        seed=42,
        early_stopping_tolerance=run_cfg.get("early_stopping_tolerance"),
    )
    if algo == "all_reduce":
        return orchestra.all_reduce(**common)

    sync  = BarrierSync()   if run_cfg.get("sync") == "barrier"  else NonBlockingSync()
    store = BlockingStore() if run_cfg.get("store") == "blocking" else WildStore()
    return orchestra.parameter_server(**common, server_addrs=server_addrs, sync=sync, store=store)


# ── Accuracy evaluation ───────────────────────────────────────────────────────


def evaluate(model_name: str, model_defs: dict, safetensors_path: Path,
             test_s: Path, test_l: Path) -> float:
    """
    Load trained weights into an equivalent PyTorch model and compute top-1 accuracy.
    Weight layout from ONO: Dense layer_N.weight = [in, out] → needs .T for PyTorch.
    Conv layer_N.weight = [filters, in_ch, kH, kW] → same as PyTorch, no transpose.
    """
    import numpy as np
    import torch
    import torch.nn as nn
    from safetensors.torch import load_file

    mdef = model_defs[model_name]
    sd   = load_file(str(safetensors_path))

    def _act(name):
        if name == "sigmoid":
            return torch.sigmoid
        if name == "softmax":
            return lambda x: torch.softmax(x, dim=-1)
        return None

    if mdef["type"] == "dense":
        dense_defs = mdef["dense"]
        sizes = [X_SIZE] + [l["size"] for l in dense_defs]

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList(
                    [nn.Linear(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)]
                )
            def forward(self, x):
                for i, layer in enumerate(self.layers):
                    x = layer(x)
                    fn = _act(dense_defs[i]["act"])
                    if fn: x = fn(x)
                return x

        net = Net()
        with torch.no_grad():
            for j, layer in enumerate(net.layers):
                layer.weight.copy_(sd[f"layer_{j}.weight"].T)
                layer.bias.copy_(sd[f"layer_{j}.bias"])
    else:
        conv_defs  = mdef["conv"]
        dense_defs = mdef["dense"]
        flat_size  = mdef["flat_size"]
        dsizes     = [flat_size] + [l["size"] for l in dense_defs]

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.convs = nn.ModuleList([
                    nn.Conv2d(in_channels=l["kernel_dim"][1], out_channels=l["kernel_dim"][0],
                              kernel_size=l["kernel_dim"][2], stride=l["stride"],
                              padding=l["padding"])
                    for l in conv_defs
                ])
                self.linears = nn.ModuleList(
                    [nn.Linear(dsizes[i], dsizes[i+1]) for i in range(len(dense_defs))]
                )
            def forward(self, x):
                x = x.view(-1, 1, 28, 28)
                for i, conv in enumerate(self.convs):
                    x = conv(x)
                    fn = _act(conv_defs[i]["act"])
                    if fn: x = fn(x)
                x = x.flatten(start_dim=1)
                for i, linear in enumerate(self.linears):
                    x = linear(x)
                    fn = _act(dense_defs[i]["act"])
                    if fn: x = fn(x)
                return x

        net = Net()
        with torch.no_grad():
            for i, conv in enumerate(net.convs):
                conv.weight.copy_(sd[f"layer_{i}.weight"])
                conv.bias.copy_(sd[f"layer_{i}.bias"])
            offset = len(conv_defs)
            for j, linear in enumerate(net.linears):
                linear.weight.copy_(sd[f"layer_{offset+j}.weight"].T)
                linear.bias.copy_(sd[f"layer_{offset+j}.bias"])

    net.eval()
    x_raw = np.fromfile(str(test_s), dtype=np.float32).reshape(-1, X_SIZE)
    y_raw = np.fromfile(str(test_l), dtype=np.float32).reshape(-1, Y_SIZE)
    x_t   = torch.tensor(x_raw)
    y_t   = torch.tensor(y_raw.argmax(axis=1), dtype=torch.long)
    with torch.no_grad():
        return (net(x_t).argmax(dim=1) == y_t).float().mean().item()


# ── Single run ────────────────────────────────────────────────────────────────


def run_single(run_def: dict, model_defs: dict) -> dict:
    model_name = run_def["model"]
    run_id = (datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
              + f"_{model_name}_run{run_def['id']}")

    result = {
        "run_id":     run_id,
        "run_num":    run_def["id"],
        "model_name": model_name,
        "workers":    run_def["workers"],
        "servers":    run_def.get("servers", 0),
        "algorithm":  run_def.get("algorithm", "parameter_server"),
        "serializer": run_def["serializer"],
        "sync":       run_def.get("sync"),
        "store":      run_def.get("store"),
        "loss_fn":    run_def.get("loss_fn", "mse"),
        "max_epochs": run_def["max_epochs"],
        "batch_size": run_def["batch_size"],
        "lr":         run_def.get("lr", 0.5),
        "early_stopping_tolerance": run_def.get("early_stopping_tolerance"),
        "docker_start_seconds": None,
        "train_seconds":        None,
        "eval_seconds":         None,
        "accuracy":             None,
        "min_accuracy":         run_def["min_accuracy"],
        "max_train_seconds":    run_def["max_train_seconds"],
        "passed":               False,
        "error":                None,
    }

    try:
        train_s, train_l, test_s, test_l = prepare_dataset()
        model    = build_model(model_name, model_defs)
        training = build_training(run_def, train_s, train_l)

        import orchestra
        t0 = time.perf_counter()
        session = orchestra.orchestrate(model, training)
        trained = session.wait()
        result["train_seconds"] = time.perf_counter() - t0

        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        sf_path = ARTIFACTS_DIR / f"{run_id}.safetensors"
        trained.save_safetensors(str(sf_path))

        t_eval = time.perf_counter()
        result["accuracy"] = evaluate(model_name, model_defs, sf_path, test_s, test_l)
        result["eval_seconds"] = time.perf_counter() - t_eval

        result["passed"] = (
            result["accuracy"] >= run_def["min_accuracy"]
            and result["train_seconds"] <= run_def["max_train_seconds"]
        )
    except Exception:
        result["error"] = traceback.format_exc()

    return result


# ── Results I/O ───────────────────────────────────────────────────────────────


def save_result(result: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(result) + "\n")


def load_historical_results() -> dict:
    """
    Load the most recent result per run_num from all JSONL files in RESULTS_DIR.
    Files are read in sorted (timestamp) order so newer sessions overwrite older ones.
    Returns {run_num: result_dict}.
    """
    by_num: dict = {}
    if not RESULTS_DIR.exists():
        return by_num
    for path in sorted(RESULTS_DIR.glob("*.jsonl")):
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "run_num" in rec and "model_name" in rec:
                        by_num[rec["run_num"]] = rec
        except OSError:
            pass
    return by_num


def print_summary(results: list):
    W = 80
    print("\n" + "─" * W)
    print(f"{'#':>3} {'MODEL':<14} {'ALG':<3} {'TOPO':<8} {'SER':<7} {'SYNC':<5} "
          f"{'TIME(s)':>8} {'ACC':>6} {'PASS':>5}")
    print("─" * W)
    for r in results:
        if "event" in r:
            continue
        t     = f"{r['train_seconds']:.1f}" if r.get("train_seconds") else " ERROR"
        a     = f"{r['accuracy']:.3f}"      if r.get("accuracy")      else "  N/A"
        p     = "yes" if r.get("passed") else "no"
        algo  = "AR" if r.get("algorithm") == "all_reduce" else "PS"
        w, s  = r.get("workers", "?"), r.get("servers", 0)
        topo  = f"{w}w" if s == 0 else f"{w}w/{s}s"
        ser   = r.get("serializer", "?")
        sync  = r.get("sync") or "—"
        sync  = sync[:5]
        print(f"{r.get('run_num','?'):>3} {r['model_name']:<14} {algo:<3} {topo:<8} "
              f"{ser:<7} {sync:<5} {t:>8} {a:>6} {p:>5}")
    print("─" * W)
    ok = sum(1 for r in results if r.get("passed"))
    total = sum(1 for r in results if "model_name" in r)
    print(f"\n{ok}/{total} passed.\n")


# ── Labels ────────────────────────────────────────────────────────────────────


def _run_label(run_def: dict) -> str:
    model = run_def.get("model") or run_def.get("model_name", "unknown")
    w     = run_def["workers"]
    s     = run_def.get("servers", 0)
    algo  = "AR" if run_def["algorithm"] == "all_reduce" else "PS"
    ser   = "+sp" if run_def["serializer"] == "sparse" else ""
    sync  = run_def.get("sync") or ""
    nb    = " nb" if sync == "nonblocking" else ""
    topo  = f"{w}w" if s == 0 else f"{w}w/{s}s"
    return f"{model}\n{algo} {topo}{ser}{nb}"


def _cmp_label(run_def: dict) -> str:
    """Compact label for per-model comparison charts (no model name)."""
    w    = run_def["workers"]
    s    = run_def.get("servers", 0)
    algo = "AR" if run_def["algorithm"] == "all_reduce" else "PS"
    ser  = "+sp" if run_def["serializer"] == "sparse" else ""
    sync = run_def.get("sync") or ""
    nb   = " nb" if sync == "nonblocking" else ""
    topo = f"{w}w" if s == 0 else f"{w}w/{s}s"
    return f"{algo} {topo}{ser}{nb}"


# ── Plots ─────────────────────────────────────────────────────────────────────


def _bar_colors(results):
    return ["#4caf50" if r.get("passed") else "#ef5350" if r.get("train_seconds") else "#9e9e9e"
            for r in results]


def _legend_patches():
    import matplotlib.patches as mpatches
    return [
        mpatches.Patch(color="#4caf50", label="pass"),
        mpatches.Patch(color="#ef5350", label="fail"),
        mpatches.Patch(color="#9e9e9e", label="error"),
    ]


def plot_main(results: list, run_ts: str):
    """Generate accuracy.png and training_time.png for all completed runs."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available — skipping plots)")
        return

    completed = [r for r in results if r.get("train_seconds") is not None and "model_name" in r]
    if not completed:
        return

    completed.sort(key=lambda r: r.get("run_num", 0))
    labels      = [f"#{r['run_num']} {_run_label(r)}" for r in completed]
    accuracies  = [r["accuracy"] if r["accuracy"] is not None else 0.0 for r in completed]
    train_times = [r["train_seconds"] for r in completed]
    colors      = _bar_colors(completed)
    x           = list(range(len(completed)))
    bar_w       = 0.55
    min_acc     = min((r.get("min_accuracy", 0.9) for r in completed if r.get("min_accuracy")), default=0.9)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Accuracy
    fig, ax = plt.subplots(figsize=(max(10, len(completed) * 1.1), 5))
    fig.suptitle(f"O.N.O MNIST Benchmark — accuracy  ({run_ts})", fontsize=12, fontweight="bold")
    bars = ax.bar(x, accuracies, width=bar_w, color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(min_acc, color="#e53935", linestyle="--", linewidth=1.3,
               label=f"min_accuracy = {min_acc:.2f}")
    for bar, val in zip(bars, accuracies):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.6)
    fig.legend(handles=_legend_patches(), loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.08), fontsize=9)
    fig.tight_layout()
    out = PLOTS_DIR / "accuracy.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  plots → {out.relative_to(BENCH_DIR)}")

    # Training time
    fig, ax = plt.subplots(figsize=(max(10, len(completed) * 1.1), 5))
    fig.suptitle(f"O.N.O MNIST Benchmark — training time  ({run_ts})", fontsize=12, fontweight="bold")
    bars = ax.bar(x, train_times, width=bar_w, color=colors, edgecolor="white", linewidth=0.8)
    offset = max(train_times) * 0.025 if train_times else 0.05
    for bar, val in zip(bars, train_times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
                f"{val:.1f}s", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Seconds")
    ax.set_ylim(0, max(max(train_times) * 1.35, 0.5))
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.6)
    fig.legend(handles=_legend_patches(), loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.08), fontsize=9)
    fig.tight_layout()
    out = PLOTS_DIR / "training_time.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  plots → {out.relative_to(BENCH_DIR)}")


def plot_comparison(results_by_num: dict, cmp_cfg: dict, run_defs_by_id: dict, run_ts: str):
    """
    Generate per-model accuracy and time charts for a comparison group.
    Produces {filename}_{model}_accuracy.png and {filename}_{model}_time.png.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    entries = []
    for rid in cmp_cfg["run_ids"]:
        rdef = run_defs_by_id.get(rid)
        r    = results_by_num.get(rid)
        if rdef is None:
            continue
        entries.append((rdef, r))

    if not entries:
        return

    # Group by model
    by_model: dict = {}
    for rdef, r in entries:
        model = rdef.get("model") or rdef.get("model_name", "unknown")
        by_model.setdefault(model, []).append((rdef, r))

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    for model_name, model_entries in sorted(by_model.items()):
        labels = [_cmp_label(rdef) for rdef, _ in model_entries]
        accs   = [(r["accuracy"] if r and r.get("accuracy") is not None else 0.0)
                  for _, r in model_entries]
        times  = [(r["train_seconds"] if r and r.get("train_seconds") else 0.0)
                  for _, r in model_entries]
        colors_list = [
            "#4caf50" if (r and r.get("passed"))
            else "#ef5350" if (r and r.get("train_seconds"))
            else "#9e9e9e"
            for _, r in model_entries
        ]
        x       = list(range(len(model_entries)))
        bar_w   = 0.55
        min_acc = min(
            (rdef.get("min_accuracy", 0.9) for rdef, _ in model_entries if rdef.get("min_accuracy")),
            default=0.9,
        )
        w_fig = max(5, len(model_entries) * 1.6)

        # Accuracy chart
        fig, ax = plt.subplots(figsize=(w_fig, 4.5))
        fig.suptitle(
            f"O.N.O — {cmp_cfg['title']}\n{model_name} · accuracy  ({run_ts})",
            fontsize=10, fontweight="bold",
        )
        bars = ax.bar(x, accs, width=bar_w, color=colors_list, edgecolor="white", linewidth=0.8)
        ax.axhline(min_acc, color="#e53935", linestyle="--", linewidth=1.3,
                   label=f"threshold {min_acc:.2f}")
        for bar, val in zip(bars, accs):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.6)
        fig.legend(handles=_legend_patches(), loc="lower center", ncol=3,
                   bbox_to_anchor=(0.5, -0.1), fontsize=9)
        fig.tight_layout()
        out = PLOTS_DIR / f"{cmp_cfg['filename']}_{model_name}_accuracy.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  plots → {out.relative_to(BENCH_DIR)}")

        # Time chart
        fig, ax = plt.subplots(figsize=(w_fig, 4.5))
        fig.suptitle(
            f"O.N.O — {cmp_cfg['title']}\n{model_name} · training time  ({run_ts})",
            fontsize=10, fontweight="bold",
        )
        bars   = ax.bar(x, times, width=bar_w, color=colors_list, edgecolor="white", linewidth=0.8)
        offset = max(times) * 0.025 if max(times, default=0) > 0 else 0.05
        for bar, val in zip(bars, times):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
                        f"{val:.1f}s", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Seconds")
        ax.set_ylim(0, max(max(times) * 1.35, 0.5) if max(times, default=0) > 0 else 0.5)
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.6)
        fig.legend(handles=_legend_patches(), loc="lower center", ncol=3,
                   bbox_to_anchor=(0.5, -0.1), fontsize=9)
        fig.tight_layout()
        out = PLOTS_DIR / f"{cmp_cfg['filename']}_{model_name}_time.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  plots → {out.relative_to(BENCH_DIR)}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="MNIST end-to-end benchmark for O.N.O")
    p.add_argument("--runs",           nargs="+", type=int, metavar="N",
                   help="Run only these numbered runs (default: all)")
    p.add_argument("--config",         type=Path,
                   default=Path(__file__).parent / "mnist_configs.json")
    p.add_argument("--output",         type=Path)
    p.add_argument("--keep-containers", action="store_true")
    p.add_argument("--rebuild",        action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    _ensure_orchestra()

    with open(args.config) as f:
        config = json.load(f)

    model_defs  = config["model_defs"]
    all_runs    = config["runs"]
    cmp_plots   = config.get("comparison_plots", [])

    # Filter by --runs if specified
    if args.runs:
        unknown = set(args.runs) - {r["id"] for r in all_runs}
        if unknown:
            sys.exit(f"ERROR: unknown run IDs: {sorted(unknown)}")
        runs_to_exec = [r for r in all_runs if r["id"] in set(args.runs)]
    else:
        runs_to_exec = list(all_runs)

    if args.output is None:
        ts = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        args.output = RESULTS_DIR / f"{ts}_benchmark.jsonl"

    # Pre-flight: check /etc/hosts
    check_hosts(runs_to_exec)

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
    except Exception:
        commit = "unknown"

    run_ids_str = " ".join(str(r["id"]) for r in runs_to_exec)
    print(f"\nO.N.O MNIST Benchmark — commit={commit}")
    print(f"runs=[{run_ids_str}]  output={args.output}\n")

    # Group by topology for Docker container reuse
    by_topo: dict = {}
    for run in runs_to_exec:
        key = (run["workers"], run.get("servers", 0))
        by_topo.setdefault(key, []).append(run)

    all_results = []

    for (workers, servers), topo_runs in sorted(by_topo.items()):
        topology_id = f"{workers}w_{servers}s"
        print(f"\n── {workers}w / {servers}s " + "─" * 50)

        docker_start_s = 0.0
        running = containers_running(workers, servers)

        try:
            ensure_hosts_for_topo(workers, servers)
            if running:
                print("  containers already running, reusing.")
            else:
                print("  starting Docker containers...")
                docker_start_s = docker_up(workers, servers, rebuild=args.rebuild)
                print(f"  Docker up in {docker_start_s:.1f}s, waiting for nodes to be ready...")
                if not wait_nodes_ready(workers, servers, timeout=60.0):
                    raise RuntimeError("nodes did not become ready within 60s")
                print(f"  nodes ready")
        except Exception as e:
            print(f"  ERROR: Docker setup failed: {e}")
            for run in topo_runs:
                result = {
                    "run_num": run["id"], "model_name": run["model"],
                    "workers": workers, "servers": servers,
                    "topology_id": topology_id, "docker_start_seconds": None,
                    "train_seconds": None, "accuracy": None,
                    "passed": False, "error": str(e),
                }
                all_results.append(result)
                save_result(result, args.output)
            continue

        save_result({"event": "topology_started", "topology_id": topology_id,
                     "docker_start_seconds": docker_start_s}, args.output)

        try:
            for i, run in enumerate(topo_runs):
                docker_restarted = False
                if i > 0:
                    is_ps = run.get("algorithm", "parameter_server") == "parameter_server"
                    prev_run = topo_runs[i - 1]
                    serializer_changed = run.get("serializer") != prev_run.get("serializer")
                    needs_restart = is_ps or serializer_changed or not wait_nodes_ready(workers, servers, timeout=30.0)
                    if needs_restart:
                        if is_ps:
                            reason = "PS inter-session reset"
                        elif serializer_changed:
                            reason = f"serializer change {prev_run.get('serializer')}→{run.get('serializer')}"
                        else:
                            reason = "nodes not ready"
                        print(f"  restarting containers ({reason})...")
                        try:
                            docker_down()
                        except Exception:
                            pass
                        docker_start_s = docker_up(workers, servers, rebuild=False)
                        docker_restarted = True
                        print(f"  Docker restarted in {docker_start_s:.1f}s")

                algo = "AR" if run["algorithm"] == "all_reduce" else "PS"
                ser  = run["serializer"]
                sync = run.get("sync") or "—"
                topo = f"{workers}w" if servers == 0 else f"{workers}w/{servers}s"
                print(f"\n  [#{run['id']}] {run['model']} × {algo} {topo}  "
                      f"ser={ser}  sync={sync}  "
                      f"ep={run['max_epochs']}  es={run.get('early_stopping_tolerance')}")

                _transient = ("deadline has elapsed", "Broken pipe", "Connection refused",
                              "connection reset", "os error 32", "os error 104")
                result = run_single(run, model_defs)
                if not result["passed"] and result.get("error") and \
                        any(t in result["error"] for t in _transient) and not docker_restarted:
                    print(f"  transient error detected, restarting containers and retrying...")
                    try:
                        docker_down()
                    except Exception:
                        pass
                    docker_start_s = docker_up(workers, servers, rebuild=False)
                    time.sleep(5)
                    docker_restarted = True
                    result = run_single(run, model_defs)
                result["docker_start_seconds"]     = docker_start_s if (i == 0 or docker_restarted) else 0.0
                result["topology_id"]              = topology_id
                result["reused_topology"]          = i > 0 and not docker_restarted
                result["docker_restarted_for_run"] = docker_restarted
                all_results.append(result)
                save_result(result, args.output)

                status = "PASS" if result["passed"] else "FAIL"
                ts_str = f"{result['train_seconds']:.1f}s" if result["train_seconds"] else "ERROR"
                ac_str = f"{result['accuracy']:.3f}"       if result["accuracy"] is not None else "N/A"
                tags   = (" [reused]"           if result["reused_topology"]          else "") + \
                         (" [fallback-restart]" if result["docker_restarted_for_run"] else "")
                print(f"  → {status}  train={ts_str}  acc={ac_str}{tags}")

        finally:
            log_label = f"{topology_id}_{datetime.datetime.now().strftime('%H-%M-%S')}"
            try:
                capture_docker_logs(workers, servers, log_label)
                print(f"  docker logs → results/docker_logs/{log_label}/")
            except Exception as ex:
                print(f"  WARNING: log capture failed: {ex}")
            if not args.keep_containers:
                print("  stopping containers...")
                try:
                    docker_down()
                except Exception as ex:
                    print(f"  WARNING: docker down failed: {ex}")

    print_summary(all_results)

    run_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    # Merge historical results with current session.
    # Current session always wins for any run it executed — other runs keep their
    # last known values from previous sessions so partial runs never blank them out.
    historical      = load_historical_results()
    current_by_num  = {r["run_num"]: r for r in all_results if "run_num" in r}
    merged_by_num   = {**historical, **current_by_num}
    merged_list     = sorted(merged_by_num.values(), key=lambda r: r.get("run_num", 0))

    plot_main(merged_list, run_ts)

    # Trigger comparison plots for any group that overlaps the current session,
    # but pass the full merged map so non-executed runs still show historical bars.
    executed_ids   = set(current_by_num)
    run_defs_by_id = {r["id"]: r for r in all_runs}

    for cmp in cmp_plots:
        if set(cmp["run_ids"]) & executed_ids:
            plot_comparison(merged_by_num, cmp, run_defs_by_id, run_ts)


if __name__ == "__main__":
    main()
