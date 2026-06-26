"""Explicit benchmark configuration: models and the four issue suites.

The configuration is declarative and fully typed — no magic strings. Layers are
dataclasses (`Conv`/`MaxPool`/`Dense`), a `Model` bundles its architecture with
its *reference* hyper-parameters, and every `Run` carries the exact
`TrainingConfig` it was launched with so the README/plots can state, for each
figure, what was trained and how.

Hyper-parameters are coupled to the model, not sprinkled across suites:

- **Nielsen** uses the canonical `network3.py` recipe from Nielsen's book
  (60 epochs, batch 10, lr 0.1) which reaches ~98.8% on MNIST. `batch_size` is
  applied *per worker* (the dataset is sharded across workers), so the small
  batch is what lets the distributed runs actually converge.
- **LeNet5** has no single canonical recipe; we use batch 64 / lr 0.05 with tanh
  activations, which reaches ~98–99%.

Speed/scalability suites do not converge — they measure throughput — so they use
a larger batch and a short budget on a subset, keeping the slow small-batch cost
confined to the convergence suites where it actually buys accuracy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from enum import Enum

X_SIZE = 784
Y_SIZE = 10
SEED = 42


# ── Typed primitives ─────────────────────────────────────────────────────────

class StrEnum(str, Enum):
    """`enum.StrEnum` backport (the project targets Python ≥ 3.10)."""

    def __str__(self) -> str:  # serialize as the plain value, not "Class.MEMBER"
        return str(self.value)


class Activation(StrEnum):
    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    SOFTMAX = "softmax"


class LayerKind(StrEnum):
    CONV = "conv"
    MAXPOOL = "maxpool"
    DENSE = "dense"


class Loss(StrEnum):
    CROSS_ENTROPY = "cross_entropy"
    MSE = "mse"


class Strategy(StrEnum):
    PARAMETER_SERVER = "parameter_server"
    ALL_REDUCE = "all_reduce"
    STRATEGY_SWITCH = "strategy_switch"


class PSVariant(StrEnum):
    """Parameter-server store/synchronizer pairing (see runner.build_training)."""

    BLOCKING = "blocking"          # BlockingStore + BarrierSync
    NON_BLOCKING = "non_blocking"  # WildStore + NonBlockingSync


# ── Layers ───────────────────────────────────────────────────────────────────

class Layer(ABC):
    @property
    @abstractmethod
    def kind(self) -> LayerKind: ...


@dataclass(frozen=True)
class Conv(Layer):
    filters: int
    kernel: int
    stride: int = 1
    padding: int = 0
    act: Activation | None = None

    @property
    def kind(self) -> LayerKind:
        return LayerKind.CONV


@dataclass(frozen=True)
class MaxPool(Layer):
    size: int
    stride: int
    padding: int = 0

    @property
    def kind(self) -> LayerKind:
        return LayerKind.MAXPOOL


@dataclass(frozen=True)
class Dense(Layer):
    out: int
    act: Activation | None = None

    @property
    def kind(self) -> LayerKind:
        return LayerKind.DENSE


# ── Training config & model ──────────────────────────────────────────────────

@dataclass(frozen=True)
class TrainingConfig:
    """Every knob a run is trained with. Suites override the model reference via
    `dataclasses.replace`, so the config behind a figure is self-describing."""

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
    layers: list[Layer]
    reference: TrainingConfig  # canonical single-process recipe for this model


# ── The two models from the issue ────────────────────────────────────────────

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
    # Canonical network3.py recipe (Nielsen, "Neural Networks and Deep Learning",
    # ch. 6): 60 epochs, mini-batch 10, lr 0.1 → ~98.8% on MNIST.
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
    # No single canonical recipe; batch 64 / lr 0.05 with tanh reaches ~98–99%.
    reference=TrainingConfig(lr=0.05, batch_size=64, max_epochs=60),
)

MODELS = {m.name: m for m in (NIELSEN, LENET5)}
ALL_MODELS = list(MODELS)


# ── Suite-wide budgets ───────────────────────────────────────────────────────

# AllReduce worker counts. The issue suggests 3/7/11; kept lighter (3/5/7) so the
# topology fits a single host. Adjust here to scale up.
AR_WORKER_SCALE = [3, 5, 7]

EARLY_STOP_TOL = 1e-4

# convergence-speed shares one FIXED budget across strategies (no early stopping)
# so loss/sec and accuracy/sec are compared on equal terms.
CONVERGENCE_SPEED_EPOCHS = 40

# Speed suites do not need to converge: a short budget on a small subset.
SPEED_EPOCHS = 8
SPEED_SUBSET = 4000

# Parameter-server topology compared across both store/sync variants.
PS_TOPOLOGY = (3, 2)  # (workers, servers)


# ── Run record ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Run:
    """A single benchmark execution. `to_record()` flattens it to the plain dict
    consumed by the runner, results and plots (the serialization boundary)."""

    suite: str
    model: Model
    strategy: Strategy
    workers: int
    servers: int
    training: TrainingConfig
    ps_variant: PSVariant | None = None  # parameter_server / strategy_switch only
    baseline: bool = False               # single-process PyTorch reference

    def to_record(self) -> dict:
        t = self.training
        return {
            "suite": self.suite,
            "model": self.model.name,
            "loss_fn": str(self.model.loss_fn),
            "strategy": str(self.strategy),
            "ps_variant": str(self.ps_variant) if self.ps_variant else None,
            "baseline": self.baseline,
            "workers": self.workers,
            "servers": self.servers,
            "lr": t.lr,
            "batch_size": t.batch_size,
            "offline_epochs": t.offline_epochs,
            "max_epochs": t.max_epochs,
            "early_stopping_tolerance": t.early_stopping_tolerance,
            "subset": t.subset,
            "eval": t.eval,
            "seed": t.seed,
            "run_key": run_key_of(self),
        }


def run_key_of(run: Run) -> str:
    t = run.training
    variant = f"|{run.ps_variant}" if run.ps_variant else ""
    base = "|baseline" if run.baseline else ""
    return (f"{run.suite}|{run.model.name}|{run.strategy}{variant}{base}"
            f"|{run.workers}w{run.servers}s"
            f"|b{t.batch_size}|o{t.offline_epochs}")


def run_key(record: dict) -> str:
    """Recompute a run key from a flat record (used on the error path)."""
    variant = f"|{record['ps_variant']}" if record.get("ps_variant") else ""
    base = "|baseline" if record.get("baseline") else ""
    return (f"{record['suite']}|{record['model']}|{record['strategy']}{variant}{base}"
            f"|{record['workers']}w{record['servers']}s"
            f"|b{record['batch_size']}|o{record['offline_epochs']}")


# ── Suite builders ───────────────────────────────────────────────────────────

# Only the blocking PS variant is benchmarked. The non-blocking variant
# (WildStore + NonBlockingSync) hangs under concurrent workers — see the tracking
# issue "Non-blocking parameter server hangs under concurrent workers". The store/sync
# wiring stays in runner.py so it can be re-enabled once the runtime bug is fixed.
PS_VARIANTS = [PSVariant.BLOCKING]


def convergence_runs(models: list[Model]) -> list[Run]:
    """Suite 1 — loss vs epoch + final accuracy. One run per strategy/topology,
    plus a single-process PyTorch baseline as a convergence reference.

    Each model trains with its *reference* recipe (Nielsen 60/10/0.1, LeNet5
    60/64/0.05) and early stopping, so the epoch budget is a ceiling. Parameter
    server runs the blocking store/sync variant (non-blocking is disabled, see
    PS_VARIANTS).
    """
    runs: list[Run] = []
    for model in models:
        cfg = replace(model.reference, eval=True,
                      early_stopping_tolerance=EARLY_STOP_TOL)
        # PyTorch single-process reference: SAME config as the distributed runs
        # (same lr/batch/epoch ceiling AND the same early-stopping rule) so the
        # convergence comparison is apples-to-apples.
        runs.append(Run("convergence", model, Strategy.ALL_REDUCE, 1, 0, cfg, baseline=True))
        for variant in PS_VARIANTS:
            runs.append(Run("convergence", model, Strategy.PARAMETER_SERVER,
                            *PS_TOPOLOGY, cfg, ps_variant=variant))
        for w in AR_WORKER_SCALE:
            runs.append(Run("convergence", model, Strategy.ALL_REDUCE, w, 0, cfg))
        runs.append(Run("convergence", model, Strategy.STRATEGY_SWITCH,
                        *PS_TOPOLOGY, cfg, ps_variant=PSVariant.BLOCKING))
    return runs


def execution_speed_runs(models: list[Model]) -> list[Run]:
    """Suite 2 — epochs/sec on all-reduce (3 workers). No convergence. Compares
    raising `offline_epochs` against raising `batch_size`."""
    runs: list[Run] = []
    for model in models:
        for offline, batch in [(0, 64), (4, 64), (0, 256)]:
            cfg = TrainingConfig(lr=model.reference.lr, batch_size=batch,
                                 max_epochs=SPEED_EPOCHS, offline_epochs=offline,
                                 subset=SPEED_SUBSET)
            runs.append(Run("execution-speed", model, Strategy.ALL_REDUCE, 3, 0, cfg))
    return runs


def convergence_speed_runs(models: list[Model]) -> list[Run]:
    """Suite 3 — loss/sec and accuracy/sec under one shared fixed budget across
    strategies (no early stopping). PS runs the blocking variant only."""
    runs: list[Run] = []
    for model in models:
        cfg = replace(model.reference, eval=True, max_epochs=CONVERGENCE_SPEED_EPOCHS,
                      early_stopping_tolerance=None)
        for variant in PS_VARIANTS:
            runs.append(Run("convergence-speed", model, Strategy.PARAMETER_SERVER,
                            *PS_TOPOLOGY, cfg, ps_variant=variant))
        runs.append(Run("convergence-speed", model, Strategy.ALL_REDUCE, 3, 0, cfg))
        runs.append(Run("convergence-speed", model, Strategy.STRATEGY_SWITCH,
                        *PS_TOPOLOGY, cfg, ps_variant=PSVariant.BLOCKING))
    return runs


def scalability_runs(models: list[Model]) -> list[Run]:
    """Suite 4 — throughput vs node count. Short budget on a subset, no
    convergence. All-reduce sweeps workers; PS sweeps workers/servers."""
    runs: list[Run] = []
    for model in models:
        speed = TrainingConfig(lr=model.reference.lr, batch_size=64,
                               max_epochs=SPEED_EPOCHS, subset=SPEED_SUBSET)
        for w in AR_WORKER_SCALE:
            runs.append(Run("scalability", model, Strategy.ALL_REDUCE, w, 0, speed))
        for w, s in [(2, 1), (3, 2)]:
            runs.append(Run("scalability", model, Strategy.PARAMETER_SERVER, w, s,
                            speed, ps_variant=PSVariant.BLOCKING))
    return runs


SUITE_BUILDERS = {
    "convergence": convergence_runs,
    "execution-speed": execution_speed_runs,
    "convergence-speed": convergence_speed_runs,
    "scalability": scalability_runs,
}

ALL_SUITES = list(SUITE_BUILDERS)


def build_runs(suites: list[str], model_names: list[str]) -> list[dict]:
    models = [MODELS[name] for name in model_names]
    records: list[dict] = []
    for suite in suites:
        for run in SUITE_BUILDERS[suite](models):
            records.append(run.to_record())
    return records
