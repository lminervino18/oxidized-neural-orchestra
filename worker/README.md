# worker

Worker-side runtime for **Oxidized Neural Orchestra (ONO)**: implements both training roles (all-reduce worker and parameter-server worker) on top of the ML engine.

## Layout

- `workers/` — the role runtimes: `all_reduce` and `parameter_server`.
- `middlewares/` — connectivity layers: the worker ring and the server cluster.
- `builder/` — assembles a worker from its spec, transports, and model.

## Key types

- `all_reduce` / `parameter_server` — the two `workers/` runtimes selected per assigned role.

## Where it fits

Compiled into the [`node`](../node/README.md) binary and selected at runtime by the spec a node receives. Uses [`comms`](../comms/README.md) for networking and [`machine_learning`](../machine_learning/README.md) for training. See the [root README](../README.md).
