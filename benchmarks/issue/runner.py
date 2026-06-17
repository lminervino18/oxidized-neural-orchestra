"""Execution layer: Docker lifecycle, dataset prep, and running a single benchmark.

Nodes are generic `node-i` containers (the orchestrator assigns server/worker
roles from `nservers`). A topology is fully described by its node count.
"""

import gzip
import os
import shutil
import struct
import subprocess
import time
import traceback
import urllib.request
from pathlib import Path

from .metrics import derive
from .models import build_orchestra_model, build_torch_ref
from .suites import X_SIZE, Y_SIZE, run_key

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DOCKER_DIR = REPO_ROOT / "docker"
COMPOSE_FILE = REPO_ROOT / "compose.yaml"
NB_DATA_DIR = REPO_ROOT / "notebooks" / "mnist" / "data"
BENCH_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BENCH_DIR / "results"
ARTIFACTS_DIR = RESULTS_DIR / "artifacts"
SUBSET_DIR = RESULTS_DIR / "data"

NODE_BASE_PORT = 40_000

MNIST_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}


# ── Dataset ──────────────────────────────────────────────────────────────────

def _dl_raw(name, url, raw_dir):
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / name
    if dest.exists():
        return dest
    gz = raw_dir / (name + ".gz")
    urllib.request.urlretrieve(url, gz)
    with gzip.open(gz, "rb") as fi, open(dest, "wb") as fo:
        shutil.copyfileobj(fi, fo)
    gz.unlink()
    return dest


def _read_images(path):
    with open(path, "rb") as f:
        _, n, rows, cols = struct.unpack(">IIII", f.read(16))
        raw = f.read(n * rows * cols)
    ppi = rows * cols
    return [[b / 255.0 for b in raw[i * ppi:(i + 1) * ppi]] for i in range(n)]


def _read_labels(path):
    with open(path, "rb") as f:
        _, n = struct.unpack(">II", f.read(8))
        return list(f.read(n))


def _write_bins(images, labels, s_path, l_path):
    s_path.parent.mkdir(parents=True, exist_ok=True)
    with open(s_path, "wb") as fs, open(l_path, "wb") as fl:
        for img, lbl in zip(images, labels):
            one_hot = [0.0] * Y_SIZE
            one_hot[lbl] = 1.0
            fs.write(struct.pack(f"{X_SIZE}f", *img))
            fl.write(struct.pack(f"{Y_SIZE}f", *one_hot))


def _copy_subset(src_s, src_l, dst_s, dst_l, n):
    SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    with open(src_s, "rb") as f:
        dst_s.write_bytes(f.read(n * X_SIZE * 4))
    with open(src_l, "rb") as f:
        dst_l.write_bytes(f.read(n * Y_SIZE * 4))


def prepare_dataset(subset=None):
    raw = NB_DATA_DIR / "mnist_raw"
    full = {
        "train_s": NB_DATA_DIR / "mnist_train_samples.bin",
        "train_l": NB_DATA_DIR / "mnist_train_labels.bin",
        "test_s": NB_DATA_DIR / "mnist_test_samples.bin",
        "test_l": NB_DATA_DIR / "mnist_test_labels.bin",
    }
    if not (full["train_s"].exists() and full["train_l"].exists()):
        _write_bins(_read_images(_dl_raw("train_images", MNIST_URLS["train_images"], raw)),
                    _read_labels(_dl_raw("train_labels", MNIST_URLS["train_labels"], raw)),
                    full["train_s"], full["train_l"])
    if not (full["test_s"].exists() and full["test_l"].exists()):
        _write_bins(_read_images(_dl_raw("test_images", MNIST_URLS["test_images"], raw)),
                    _read_labels(_dl_raw("test_labels", MNIST_URLS["test_labels"], raw)),
                    full["test_s"], full["test_l"])

    train_s, train_l = full["train_s"], full["train_l"]
    if subset:
        full_n = full["train_s"].stat().st_size // (X_SIZE * 4)
        if subset < full_n:
            sub_s = SUBSET_DIR / f"mnist_train_{subset}_samples.bin"
            sub_l = SUBSET_DIR / f"mnist_train_{subset}_labels.bin"
            if not (sub_s.exists() and sub_l.exists()):
                _copy_subset(full["train_s"], full["train_l"], sub_s, sub_l, subset)
            train_s, train_l = sub_s, sub_l
    return train_s, train_l, full["test_s"], full["test_l"]


# ── Docker / hosts ─────────────────────────────────────────────────────────────

def node_addrs(n):
    return [f"node-{i}:{NODE_BASE_PORT + i}" for i in range(n)]


def nodes_for(run):
    return run["workers"] + run["servers"]


def _hosts_ok(n):
    try:
        content = Path("/etc/hosts").read_text()
        return all(f"node-{i}" in content for i in range(n))
    except OSError:
        return False


def check_hosts(max_nodes):
    if _hosts_ok(max_nodes):
        return
    missing = [f"127.0.0.1 node-{i}" for i in range(max_nodes)]
    raise SystemExit(
        "ERROR: /etc/hosts is missing node entries.\nFix with:\n"
        f'  sudo bash -c "printf \'%s\\n\' {" ".join(repr(m) for m in missing)} >> /etc/hosts"'
    )


def _gen_compose(nodes, release=True):
    env = {**os.environ, "NODES": str(nodes), "RELEASE": str(release).lower()}
    subprocess.run(["python3", str(DOCKER_DIR / "gen_compose.py")], env=env, check=True, cwd=str(REPO_ROOT))


def docker_up(nodes, rebuild=False, release=True):
    _gen_compose(nodes, release)
    if rebuild:
        subprocess.run(["docker", "compose", "-f", str(COMPOSE_FILE), "build", "--no-cache"], check=True)
    subprocess.run(["docker", "compose", "-f", str(COMPOSE_FILE), "up", "--build", "-d", "--remove-orphans"],
                   check=True)


def docker_down():
    subprocess.run(["docker", "compose", "-f", str(COMPOSE_FILE), "down"], check=True, capture_output=True)
    time.sleep(2)


def _containers_running(nodes):
    for i in range(nodes):
        r = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", f"node-{i}"],
                           capture_output=True, text=True)
        if r.returncode != 0 or r.stdout.strip() != "true":
            return False
    return True


def wait_nodes_ready(nodes, timeout=60.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _containers_running(nodes):
            return True
        time.sleep(1.0)
    return False


def docker_cleanup():
    """Tear down any leftover compose project so a run starts from a clean slate."""
    if not COMPOSE_FILE.exists():
        return
    try:
        subprocess.run(["docker", "compose", "-f", str(COMPOSE_FILE), "down", "--remove-orphans"],
                       check=False, capture_output=True)
    except Exception:
        pass


def capture_logs(nodes, label):
    log_dir = RESULTS_DIR / "docker_logs" / label
    log_dir.mkdir(parents=True, exist_ok=True)
    for i in range(nodes):
        r = subprocess.run(["docker", "logs", f"node-{i}"], capture_output=True, text=True)
        (log_dir / f"node-{i}.log").write_text(r.stdout + r.stderr)


# ── Training / eval ────────────────────────────────────────────────────────────

def build_training(run, train_s, train_l):
    import orchestra
    from orchestra.datasets import LocalDataset
    from orchestra.loss_fns import CrossEntropy, Mse
    from orchestra.optimizers import GradientDescent
    from orchestra.store import BlockingStore
    from orchestra.sync import BarrierSync

    addrs = node_addrs(nodes_for(run))
    loss_fn = CrossEntropy() if run["loss_fn"] == "cross_entropy" else Mse()
    common = dict(
        addrs=addrs,
        dataset=LocalDataset(str(train_s), str(train_l), x_size=X_SIZE, y_size=Y_SIZE),
        optimizer=GradientDescent(lr=run["lr"]),
        loss_fn=loss_fn,
        max_epochs=run["max_epochs"],
        batch_size=run["batch_size"],
        offline_epochs=run["offline_epochs"],
        seed=run["seed"],
        early_stopping_tolerance=run.get("early_stopping_tolerance"),
    )
    strategy = run["strategy"]
    if strategy == "all_reduce":
        return orchestra.all_reduce(**common)
    extra = dict(nservers=run["servers"], sync=BarrierSync(), store=BlockingStore())
    if strategy == "parameter_server":
        return orchestra.parameter_server(**common, **extra)
    return orchestra.strategy_switch(**common, **extra)


def evaluate(model_name, sf_path, test_s, test_l):
    import numpy as np
    import torch
    from safetensors.torch import load_file

    net = build_torch_ref(model_name, load_file(str(sf_path)))
    x = np.fromfile(str(test_s), dtype=np.float32).reshape(-1, X_SIZE)
    y = np.fromfile(str(test_l), dtype=np.float32).reshape(-1, Y_SIZE)
    xt = torch.tensor(x)
    yt = torch.tensor(y.argmax(axis=1), dtype=torch.long)
    with torch.no_grad():
        return (net(xt).argmax(dim=1) == yt).float().mean().item()


def run_single(run):
    keep = ("suite", "model", "strategy", "workers", "servers",
            "batch_size", "offline_epochs", "max_epochs", "lr")
    result = {k: run[k] for k in keep}
    result["run_key"] = run_key(run)
    result.update(train_seconds=None, loss_history=None, accuracy=None, error=None)

    try:
        train_s, train_l, test_s, test_l = prepare_dataset(run.get("subset"))
        model = build_orchestra_model(run["model"])
        training = build_training(run, train_s, train_l)

        import orchestra
        t0 = time.perf_counter()
        trained = orchestra.orchestrate(model, training).wait()
        result["train_seconds"] = time.perf_counter() - t0
        result["loss_history"] = trained.loss_history()

        if run.get("eval"):
            ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
            sf = ARTIFACTS_DIR / (result["run_key"].replace("|", "_").replace("/", "-") + ".safetensors")
            trained.save_safetensors(str(sf))
            result["accuracy"] = evaluate(run["model"], sf, test_s, test_l)
    except Exception:
        result["error"] = traceback.format_exc()

    return derive(result)
