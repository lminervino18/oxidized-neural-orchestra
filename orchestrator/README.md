# orchestrator

Drives a distributed run in **Oxidized Neural Orchestra (ONO)**: owns the canonical config schema, connects to nodes, assigns roles, and streams training events back to the caller.

## Layout

- `configs/` — the canonical config schema (`ModelConfig`, `TrainingConfig`, `AlgorithmConfig`) plus validation and adaptation.
- `dataset_format/` — detects delimited formats and converts datasets to packed `f32` binary.
- `sessions/` — the live `Session`, its handles, and the `TrainingEvent` stream.
- internal: `calculator`, `error`.

## Key types

- `Session` / `TrainedModel` — the active run and its final result.
- `TrainingEvent` / `StopReason` / `CancelHandle` — event stream and lifecycle control.
- `train(model, training)` — entry point that validates configs, connects to nodes, and returns a `Session`.

## Where it fits

Uses [`comms`](../comms/README.md) to connect nodes and assign roles; consumed by `orchestui` and `orchestra-py` to launch and monitor runs. See the [root README](../README.md).
