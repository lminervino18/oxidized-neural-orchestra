import os

import orchestra
from orchestra import Sequential, orchestrate
from orchestra.activations import Sigmoid
from orchestra.arch import Dense
from orchestra.datasets import LocalDataset
from orchestra.initialization import Kaiming
from orchestra.loss_fns import Mse
from orchestra.optimizers import GradientDescent
from orchestra.serializer import SparseSerializer
from orchestra.store import WildStore
from orchestra.sync import NonBlockingSync


def _make_addrs(n: int, base_port: int, host: str) -> list[str]:
    return [f"{host}-{i}:{base_port + i}" for i in range(n)]


_nodes = int(os.environ.get("NODES", "6"))
_servers = int(os.environ.get("SERVERS", "3"))

ADDRS = _make_addrs(_nodes, 40000, "node")

SAMPLES_PATH = os.environ.get("SAMPLES_PATH", "data/mnist_train_samples.bin")
LABELS_PATH = os.environ.get("LABELS_PATH", "data/mnist_train_labels.bin")
SAFETENSORS_PATH = os.environ.get("SAFETENSORS_PATH", "model.safetensors")

EARLY_STOPPING_TOLERANCE: float | None = float(t) if (t := os.environ.get("EARLY_STOPPING_TOLERANCE")) else None


def main() -> None:
    print(f"node addrs: {ADDRS}")
    print(f"servers: {_servers}")
    print(f"samples: {SAMPLES_PATH}")
    print(f"labels: {LABELS_PATH}")
    if EARLY_STOPPING_TOLERANCE is not None:
        print(f"early stopping tolerance: {EARLY_STOPPING_TOLERANCE:.2e}")
    else:
        print("early stopping: disabled")

    print("\nbuilding model...")
    model = Sequential([
        Dense(128, Kaiming(), Sigmoid()),
        Dense(64, Kaiming(), Sigmoid()),
        Dense(10, Kaiming()),
    ])

    print("building training config...")
    training = orchestra.parameter_server(
        addrs=ADDRS,
        nservers=_servers,
        dataset=LocalDataset(SAMPLES_PATH, LABELS_PATH, x_size=784, y_size=10),
        optimizer=GradientDescent(lr=0.01),
        loss_fn=Mse(),
        sync=NonBlockingSync(),
        store=WildStore(),
        serializer=SparseSerializer(r=0.9),
        max_epochs=50,
        batch_size=256,
        offline_epochs=1,
        seed=42,
        early_stopping_tolerance=EARLY_STOPPING_TOLERANCE,
    )

    print("\nstarting training session...")
    session = orchestrate(model, training)
    trained = session.wait()

    print("\nsaving model...")
    trained.save_safetensors(SAFETENSORS_PATH)
    print(f"saved model to {SAFETENSORS_PATH}")


if __name__ == "__main__":
    main()