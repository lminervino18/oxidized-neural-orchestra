#!/bin/bash

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ORCHESTUI_DIR="$ROOT/orchestui"

RELEASE="--release"

# Node-agnostic setup: every node is an identical container and the orchestrator
# assigns server/worker roles at runtime. We only need the total node count,
# which is the number of addresses listed in training.json (the same file the
# TUI loads by default, from the orchestui/ directory).
NNODES=$(python3 -c "import json; print(len(json.load(open('$ORCHESTUI_DIR/training.json'))['addrs']))")

# Bring up all nodes via Docker.
python3 "$ROOT/docker/compose_up.py" \
  --nodes "$NNODES" \
  $RELEASE

sleep 2

# Open Docker logs in a new terminal
setsid gnome-terminal --title="docker-logs" -- bash -c "
  docker compose -f $ROOT/compose.yaml logs -f
  exec bash
" &

sleep 0.5

# Run the TUI from the repo root; by default it loads orchestui/model.json and
# orchestui/training.json (relative to the root).
cd "$ROOT"
RUST_LOG=info cargo run ${RELEASE:+--release} -p orchestui 2>/tmp/ono-orchestrator.log