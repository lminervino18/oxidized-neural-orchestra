# machine_learning

The neural-network engine of **Oxidized Neural Orchestra (ONO)**: layers, models, initializers, optimizers, datasets, and the training loop, all decoupled from any distribution strategy.

## Layout

- `arch/` — network architecture building blocks (layers, activations).
- `models/` — assembled models and forward/backward passes.
- `initialization/` — weight/bias initializers.
- `optimization/` — optimizers and update rules.
- `datasets/` — dataset loading and batching.
- `param_manager/` — parameter storage and gradient bookkeeping.
- `training/` — the `Trainer` and training-loop orchestration.
- `error/` — `MlErr` and the crate `Result` alias.

## Key types

- `Trainer` — drives the local training loop over a model and dataset.
- `MlErr` / `Result` — crate-wide error type and result alias.

## Where it fits

Consumed by [`worker`](../worker/README.md) (per-role training) and `orchestra-py` (Python bindings). Pure compute — it holds no networking; distribution is layered on top by `worker`. See the [root README](../README.md).
