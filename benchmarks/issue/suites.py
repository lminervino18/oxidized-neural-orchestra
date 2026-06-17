"""Explicit benchmark configuration: models and the four issue suites.

Every run declares its strategy, topology and hyper-parameters. The only
variable that changes within a suite is the architecture (worker/server count)
unless the suite explicitly compares another knob (see execution-speed).
"""

X_SIZE = 784
Y_SIZE = 10

# Models — only the two defined in the issue.
# Layer dims are propagated automatically in models.py from `input_dim`.
MODELS = {
    "nielsen": {
        "input_dim": (1, 28, 28),
        "loss_fn": "cross_entropy",
        "lr": 0.1,
        "layers": [
            {"type": "conv", "filters": 20, "kernel": 5, "stride": 1, "padding": 0},
            {"type": "maxpool", "size": 2, "stride": 2, "padding": 0},
            {"type": "dense", "out": 100, "act": "sigmoid"},
            {"type": "dense", "out": Y_SIZE, "act": "softmax"},
        ],
    },
    "lenet5": {
        "input_dim": (1, 28, 28),
        "loss_fn": "cross_entropy",
        "lr": 0.05,
        "layers": [
            {"type": "conv", "filters": 6, "kernel": 5, "stride": 1, "padding": 2, "act": "tanh"},
            {"type": "maxpool", "size": 2, "stride": 2, "padding": 0},
            {"type": "conv", "filters": 16, "kernel": 5, "stride": 1, "padding": 0, "act": "tanh"},
            {"type": "maxpool", "size": 2, "stride": 2, "padding": 0},
            {"type": "dense", "out": 120, "act": "tanh"},
            {"type": "dense", "out": 84, "act": "tanh"},
            {"type": "dense", "out": Y_SIZE, "act": "softmax"},
        ],
    },
}

ALL_MODELS = list(MODELS)

# AllReduce worker counts. The issue suggests 3/7/11; kept lighter (3/5/7) so the
# topology fits a single host. Adjust here to scale up.
AR_WORKER_SCALE = [3, 5, 7]

# Generous epoch ceiling; early stopping ends a run as soon as it converges,
# so convergence runs rarely use the full budget.
CONVERGENCE_EPOCHS = 60
EARLY_STOP_TOL = 1e-4

# convergence-speed shares one FIXED budget across strategies (no early stopping)
# so loss/sec and accuracy/sec are compared on equal terms.
CONVERGENCE_SPEED_EPOCHS = 40

# Speed suites do not need to converge: a short budget on a small subset.
SPEED_EPOCHS = 8
SPEED_SUBSET = 4000

SEED = 42


def _base(model):
    m = MODELS[model]
    return {"model": model, "loss_fn": m["loss_fn"], "lr": m["lr"], "seed": SEED}


def convergence_runs(models):
    """Suite 1 — loss vs epoch + final accuracy. One run per strategy/topology.

    Early stopping ends each run when it converges, so the epoch budget is a
    ceiling rather than a fixed cost.
    """
    runs = []
    base = dict(suite="convergence", eval=True, batch_size=64, offline_epochs=0,
                max_epochs=CONVERGENCE_EPOCHS, early_stopping_tolerance=EARLY_STOP_TOL)
    for model in models:
        runs.append({**_base(model), **base,
                     "strategy": "parameter_server", "workers": 3, "servers": 2})
        for w in AR_WORKER_SCALE:
            runs.append({**_base(model), **base,
                         "strategy": "all_reduce", "workers": w, "servers": 0})
        runs.append({**_base(model), **base,
                     "strategy": "strategy_switch", "workers": 3, "servers": 2})
    return runs


def execution_speed_runs(models):
    """Suite 2 — epochs/sec. No convergence. Minimal offline_epochs vs batch_size sweep."""
    runs = []
    for model in models:
        for offline, batch in [(0, 64), (4, 64), (0, 256)]:
            runs.append({**_base(model), "suite": "execution-speed", "eval": False,
                         "strategy": "all_reduce", "workers": 3, "servers": 0,
                         "batch_size": batch, "offline_epochs": offline,
                         "max_epochs": SPEED_EPOCHS, "subset": SPEED_SUBSET})
    return runs


def convergence_speed_runs(models):
    """Suite 3 — loss/sec and accuracy/sec under one shared budget across strategies."""
    runs = []
    for model in models:
        common = dict(batch_size=64, offline_epochs=0, max_epochs=CONVERGENCE_SPEED_EPOCHS,
                      eval=True, suite="convergence-speed")
        runs.append({**_base(model), **common, "strategy": "parameter_server", "workers": 3, "servers": 2})
        runs.append({**_base(model), **common, "strategy": "all_reduce", "workers": 3, "servers": 0})
        runs.append({**_base(model), **common, "strategy": "strategy_switch", "workers": 3, "servers": 2})
    return runs


def scalability_runs(models):
    """Suite 4 — throughput vs node count. Short budget, no convergence re-test."""
    runs = []
    for model in models:
        for w in AR_WORKER_SCALE:
            runs.append({**_base(model), "suite": "scalability", "eval": False,
                         "strategy": "all_reduce", "workers": w, "servers": 0,
                         "batch_size": 64, "offline_epochs": 0,
                         "max_epochs": SPEED_EPOCHS, "subset": SPEED_SUBSET})
        for w, s in [(2, 1), (3, 2)]:
            runs.append({**_base(model), "suite": "scalability", "eval": False,
                         "strategy": "parameter_server", "workers": w, "servers": s,
                         "batch_size": 64, "offline_epochs": 0,
                         "max_epochs": SPEED_EPOCHS, "subset": SPEED_SUBSET})
    return runs


SUITE_BUILDERS = {
    "convergence": convergence_runs,
    "execution-speed": execution_speed_runs,
    "convergence-speed": convergence_speed_runs,
    "scalability": scalability_runs,
}

ALL_SUITES = list(SUITE_BUILDERS)


def build_runs(suites, models):
    runs = []
    for suite in suites:
        runs.extend(SUITE_BUILDERS[suite](models))
    return runs


def run_key(run):
    return (f"{run['suite']}|{run['model']}|{run['strategy']}"
            f"|{run['workers']}w{run['servers']}s"
            f"|b{run['batch_size']}|o{run['offline_epochs']}")
