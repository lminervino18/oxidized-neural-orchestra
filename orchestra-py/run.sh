#!/bin/bash

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ORCHESTRA_PY_DIR="$ROOT/orchestra-py"

# Parse --release flag
RELEASE=""
for arg in "$@"; do
  [ "$arg" = "--release" ] && RELEASE="--release"
done

# Read worker and server counts from env or defaults
WORKERS="${WORKERS:-3}"
SERVERS="${SERVERS:-3}"

# Bring up workers and servers via Docker
python3 "$ROOT/docker/compose_up.py" \
  --workers "$WORKERS" \
  --servers "$SERVERS" \
  $RELEASE

sleep 5

# Open Docker logs in a new terminal
setsid gnome-terminal --title="docker-logs" -- bash -c "
  docker compose -f $ROOT/compose.yaml logs -f
  exec bash
" &

sleep 0.5

# Run orchestra-py local.py with Docker hostnames and dataset path
cd "$ROOT"
WORKERS="$WORKERS" \
SERVERS="$SERVERS" \
DATASET_PATH="data/mnist_train.bin" \
RUST_LOG=info \
python3 "$ORCHESTRA_PY_DIR/local.py"