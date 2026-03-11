import sys
import os

import orchestra
from orchestra import Sequential, orchestrate
from orchestra.arch import Dense
from orchestra.activations import Sigmoid
from orchestra.initialization import Kaiming
from orchestra.datasets import InlineDataset, LocalDataset
from orchestra.optimizers import GradientDescent
from orchestra.sync import BarrierSync
from orchestra.store import BlockingStore


def _make_addrs(n: int, base_port: int, host: str = "worker") -> list[str]:
    return [f"{host}-{i}:{base_port + i}" for i in range(n)]


_workers = int(os.environ.get("WORKERS", "3"))
_servers = int(os.environ.get("SERVERS", "2"))

WORKER_ADDRS = os.environ.get(
    "WORKER_ADDRS",
    ",".join(_make_addrs(_workers, 50000, "127.0.0.1")),
).split(",")

SERVER_ADDRS = os.environ.get(
    "SERVER_ADDRS",
    ",".join(_make_addrs(_servers, 40000, "127.0.0.1")),
).split(",")

# In Docker, use container hostnames instead of localhost.
if os.environ.get("WORKERS"):
    WORKER_ADDRS = _make_addrs(_workers, 50000, "worker")
    SERVER_ADDRS = _make_addrs(_servers, 40000, "server")

# Inline fallback dataset for environments without a local file.
_INLINE_DATA = [
    1.0, 2.0,
    2.0, 4.0,
    3.0, 6.0,
    4.0, 8.0,
    5.0, 10.0,
    6.0, 12.0,
    7.0, 14.0,
    8.0, 16.0,
]


def _build_dataset():
    """
    Returns a dataset instance.

    Resolution order:
    1. DATASET_PATH env var — used in Docker via volume mount.
    2. data/dataset relative to cwd — used for local development.
    3. Inline fallback — used when no file is available at all.
    """
    dataset_path = os.environ.get("DATASET_PATH")
    if not dataset_path and os.path.exists("data/dataset"):
        dataset_path = "data/dataset"
    if dataset_path:
        return LocalDataset(dataset_path, x_size=2, y_size=1)
    return InlineDataset(_INLINE_DATA, x_size=1, y_size=1)


def print_params(params: list[float], output_sizes: list[int], input_size: int) -> None:
    print(f"trained parameters ({len(params)} total):")
    offset = 0
    prev = input_size
    for layer_i, out in enumerate(output_sizes):
        w_count = prev * out
        b_count = out
        weights = params[offset : offset + w_count]
        biases = params[offset + w_count : offset + w_count + b_count]
        print(f"\n  layer {layer_i}  ({prev}x{out})")
        print(f"    weights: {[round(w, 4) for w in weights]}")
        print(f"    biases:  {[round(b, 4) for b in biases]}")
        offset += w_count + b_count
        prev = out


def main() -> None:
    print(f"worker addrs: {WORKER_ADDRS}")
    print(f"server addrs: {SERVER_ADDRS}")

    dataset = _build_dataset()
    print(f"dataset: {dataset.__class__.__name__}")

    print("\nbuilding model...")
    model = Sequential([
        Dense(8, Kaiming(), Sigmoid()),
        Dense(4, Kaiming(), Sigmoid()),
        Dense(1, Kaiming()),
    ])

    print("building training config...")
    training = orchestra.parameter_server(
        worker_addrs=WORKER_ADDRS,
        server_addrs=SERVER_ADDRS,
        dataset=dataset,
        optimizer=GradientDescent(lr=0.01),
        sync=BarrierSync(),
        store=BlockingStore(),
        max_epochs=100,
        batch_size=4,
        seed=42,
    )

    print("\nstarting training session...")
    session = orchestrate(model, training)

    print("waiting for training to complete...")
    try:
        trained = session.wait()
        params = trained.weights()
        print_params(params, [8, 4, 1], input_size=1)
        trained.save("weights.csv", output_sizes=[8, 4, 1], input_size=1)
        print("\nweights saved to weights.csv")
    except RuntimeError as e:
        print(f"training failed: {e}", file=sys.stderr)
        sys.exit(1)

    print("done.")


if __name__ == "__main__":
    main()