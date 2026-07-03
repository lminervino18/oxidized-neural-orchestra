# comms

The networking backbone of **Oxidized Neural Orchestra (ONO)**: transports, the connection handshake, node handles, and the wire protocol every runtime crate speaks over.

## Layout

- `floats/` — packed and sparse `f32` payload encoding for gradients and parameters.
- `protocol/` — wire `specs`, message types, and the `Entity` role tags.
- `share_dataset/` — streaming a dataset from the orchestrator to nodes.
- internal: `clusters`, `codec`, `connection`, `handles`, `sparse`, `transport`, `utils`.

## Key types

- `NetRtp` / `Stp` — reliable (RTP) and simple transport layers; `TransportLayer` is the shared trait.
- `Acceptor` / `Connector` — the two sides of the connection handshake.
- `NodeHandle` / `OrchHandle` / `WorkerHandle` / `ParamServerHandle` — typed handles for talking to each role.
- `ParamServerCluster` — a handle group over a set of parameter servers.

## Where it fits

Every runtime crate ([`worker`](../worker/README.md), [`parameter_server`](../parameter_server/README.md), [`orchestrator`](../orchestrator/README.md), [`node`](../node/README.md)) uses `comms` to connect and exchange gradients and parameters. See the [root README](../README.md).
