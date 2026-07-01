# orchestui

`orchestui` is an interactive [ratatui](https://ratatui.rs) TUI for **oxidized-neural-orchestra**. It lets you
configure, launch and monitor a distributed training run from a single terminal
application. It embeds and drives the `orchestrator` crate directly: from the UI
you point it at two JSON config files, it connects to the `node` processes
listening on the addresses you provide, assigns server/worker roles, and streams
live loss, per-worker progress and a topology view back into the dashboard.

In other words, orchestui is the **orchestrator side** of the system — the
control plane. The compute nodes (`node` processes) are raised separately, either
by hand or via Docker.

---

## Build & run

From the repo root:

```bash
cargo run -p orchestui --release
```

On startup the TUI prompts for the paths to two JSON files: `model.json` (the
network architecture) and `training.json` (the distributed training setup).
Leave a prompt blank to use `model.json` / `training.json` from the current
working directory.

> The nodes are a prerequisite: orchestui connects out to the addresses listed
> in `training.json`, so those `node` processes must already be listening (see
> [How it fits together](#how-it-fits-together)). If nothing answers, the TUI
> shows a "Failed to Start Training" screen.

### End-to-end helper

[`run.sh`](./run.sh) brings the whole thing up in one shot:

```bash
./orchestui/run.sh
```

It reads the number of `addrs` from `training.json`, launches that many identical
Docker nodes via `docker/compose_up.py`, opens the Docker logs in a new terminal,
and then starts the TUI. Every node is an identical container; the orchestrator
assigns the server/worker roles at runtime.

---

## Using the interface

The app is a small state machine with three screens.

### Main menu

| Key | Action |
|---|---|
| `↑` / `↓` or `k` / `j` | Move selection |
| `Enter` | Select (`Start Training`, `Repository`, `Quit`) |
| `q` | Quit |

`Repository` opens the project page in your browser.

### Configuration screen

A two-step wizard: step 1 asks for the `model.json` path, step 2 for the
`training.json` path.

| Key | Action |
|---|---|
| type / `Backspace` | Edit the path |
| `←` / `→` | Move the cursor |
| `Tab` or `→` (at end) | Accept the ghost-text suggestion (the default path) |
| `?` | Show an inline config example for the current file |
| `Enter` | Confirm and advance / load the configs |
| `Esc` | Back (to previous step, or to the menu) |

A blank field falls back to `model.json` / `training.json` in the current
directory. If either file is missing or fails to parse, an "Invalid
Configuration" screen shows the reason; press any key to retry or `q` / `Esc` to
return to the menu.

### Training dashboard

Once the configs load, the session starts in a background thread (converting the
dataset if needed, then connecting) and the dashboard appears: a live per-worker
loss chart, a workers table, a parameters panel (after completion) and a scrolling
log. The header shows the algorithm, optimizer, epoch progress and elapsed time.

| Key | Action |
|---|---|
| `v` | Toggle between the **dashboard** and the full-screen **topology** view |
| `←` / `→` | Switch the focused worker (charts/table highlight) |
| `x` | Stop training (asks for confirmation; only while running) |
| `s` | Save the trained model to a `.safetensors` file (only after it finishes) |
| `q` / `Esc` | Back to the menu (asks for confirmation while running) |

### Topology view

Pressing `v` switches to a full-canvas visualization of the nodes as workers and
servers. It updates live — with the `strategy_switch` algorithm you can watch a
worker upgrade into a parameter server mid-run.

---

## Configuration

Two files drive a run: `model.json` (the network architecture) and
`training.json` (the distributed-training setup). These are the canonical
`orchestrator` configs (`orchestrator::configs`); the TUI parses them verbatim.

The full field-by-field reference — every layer, initializer, activation,
algorithm, optimizer, dataset source, and optional field — lives in
**[`../docs/config-schema.md`](../docs/config-schema.md)**. Working example files
also ship with the crate: [`model.example`](./model.example) and
[`training.example`](./training.example).

### `model.json` (minimal)

```json
{
  "layers": [
    { "dense": { "output_size": 8, "init": "kaiming", "act_fn": { "sigmoid": { "amp": 1.0 } } } },
    { "dense": { "output_size": 1, "init": "kaiming" } }
  ]
}
```

### `training.json` (minimal)

```json
{
  "addrs": ["127.0.0.1:50000", "127.0.0.1:50001", "127.0.0.1:50002"],
  "algorithm": { "parameter_server": { "nservers": 1, "synchronizer": "barrier", "store": "blocking" } },
  "dataset": {
    "src": { "local": { "samples_path": "data/samples.csv", "labels_path": "data/labels.csv" } },
    "x_size": 2,
    "y_size": 1
  },
  "optimizer": { "gradient_descent": { "lr": 0.01 } },
  "loss_fn": "mse",
  "batch_size": 4,
  "max_epochs": 500,
  "offline_epochs": 0
}
```

`algorithm` can be `"all_reduce"`, `parameter_server`, or `strategy_switch` — with
the `strategy_switch` algorithm you can watch a worker upgrade into a parameter
server live in the topology view. Optional fields (`seed`, `serializer`,
`early_stopping`) and every allowed value are documented in
[`../docs/config-schema.md`](../docs/config-schema.md).

---

## How it fits together

orchestui is the control plane. It never listens for connections itself — it
**connects out** to the `node` processes at the `addrs` from `training.json`.
Those nodes are identical and role-agnostic; orchestui (via the `orchestrator`
crate) tells each one at runtime whether it will act as a worker or a parameter
server, based on the chosen `algorithm` and `nservers`.

So a typical run is:

1. Raise the compute nodes — manually, or via [`run.sh`](./run.sh) /
   `docker/compose_up.py`, which spin up one identical container per address in
   `training.json`.
2. Start orchestui, point it at `model.json` and `training.json`.
3. orchestui connects to every `addr`, assigns roles, and drives training —
   streaming loss, per-worker progress and topology back into the dashboard.

Because roles are assigned at runtime, `training.json` only needs the total node
count (the length of `addrs`) plus how many of them should be servers
(`nservers`). There is no separate worker/server address list.
