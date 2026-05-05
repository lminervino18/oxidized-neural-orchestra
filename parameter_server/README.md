# parameter_server

Rust library providing parameter storage, synchronization, and server logic for distributed training. Embedded in the `node` binary — not deployed as a standalone service.

## Structure

| Module | Contents |
|---|---|
| `service/` | `ParameterServer` runtime, `ServerBuilder` |
| `storage/` | `BlockingStore`, `WildStore`, `StoreHandle` |
| `synchronization/` | `BarrierSync`, `NoBlockingSync`, `Synchronizer` trait |

## Storage strategies

| Type | Description |
|---|---|
| `BlockingStore` | Double-buffered sharded store. Gradient accumulation and parameter reads are protected by an atomic update lock. Safe under concurrent workers. |
| `WildStore` | Lock-free sharded store. Gradients are applied in-place without coordination. Higher throughput; may return stale parameters under high concurrency. |

Both stores shard parameters across CPU cores and use Rayon for parallel reads and writes.

## Synchronization strategies

| Type | Description |
|---|---|
| `BarrierSync` | Workers synchronize at each epoch before the server applies the accumulated gradient. Uses a `DrainableBarrier` — disconnecting workers drain the barrier automatically on drop, unblocking remaining workers without deadlock. |
| `NoBlockingSync` | Gradients are applied immediately as they arrive. No coordination between workers. |

## Env (set on the `node` process)

```bash
HOST=0.0.0.0   # Bind address.
PORT=8765      # Port to listen on.
RUST_LOG=info  # Log level.
```
