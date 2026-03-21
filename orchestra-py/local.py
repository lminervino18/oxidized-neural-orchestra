import sys
import os

import orchestra
from orchestra import Sequential, orchestrate
from orchestra.arch import Dense
from orchestra.activations import Sigmoid
from orchestra.initialization import Kaiming
from orchestra.datasets import LocalDataset
from orchestra.optimizers import GradientDescent
from orchestra.sync import NonBlockingSync
from orchestra.store import WildStore


def _make_addrs(n: int, base_port: int, host: str) -> list[str]:
    return [f"{host}-{i}:{base_port + i}" for i in range(n)]


_workers = int(os.environ.get("WORKERS", "3"))
_servers = int(os.environ.get("SERVERS", "3"))

WORKER_ADDRS = _make_addrs(_workers, 50000, "worker")
SERVER_ADDRS = _make_addrs(_servers, 40000, "server")

DATASET_PATH = os.environ.get("DATASET_PATH", "data/mnist_train.bin")
SAFETENSORS_PATH = os.environ.get("SAFETENSORS_PATH", "model.safetensors")


def main() -> None:
    print(f"worker addrs: {WORKER_ADDRS}")
    print(f"server addrs: {SERVER_ADDRS}")
    print(f"dataset: {DATASET_PATH}")

    print("\nbuilding model...")
    model = Sequential([
        Dense(128, Kaiming(), Sigmoid()),
        Dense(64,  Kaiming(), Sigmoid()),
        Dense(10,  Kaiming()),
    ])

    print("building training config...")
    training = orchestra.parameter_server(
        worker_addrs=WORKER_ADDRS,
        server_addrs=SERVER_ADDRS,
        dataset=LocalDataset(DATASET_PATH, x_size=784, y_size=10),
        optimizer=GradientDescent(lr=0.01),
        sync=NonBlockingSync(),
        store=WildStore(),
        max_epochs=50,
        batch_size=256,
        offline_epochs=1,
        seed=42,
    )

    print("\nstarting training session...")
    session = orchestrate(model, training)

    print("waiting for training to complete...")
    try:
        trained = session.wait()
        print(f"training complete — {len(trained.weights())} parameters")
        trained.save_safetensors(SAFETENSORS_PATH)
        print(f"model saved to {SAFETENSORS_PATH}")
    except RuntimeError as e:
        print(f"training failed: {e}", file=sys.stderr)
        sys.exit(1)

    print("done.")


if __name__ == "__main__":
    main()