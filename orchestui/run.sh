#!/bin/bash

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ORCHESTUI_DIR="$ROOT/orchestui"

# Kill any leftover processes by port
ALL_PORTS=$(python3 -c "
import json
d = json.load(open('$ORCHESTUI_DIR/training.json'))
addrs = d['algorithm']['parameter_server']['server_addrs'] + d['worker_addrs']
print(' '.join(a.split(':')[1] for a in addrs))
")

for port in $ALL_PORTS; do
  pid=$(lsof -t -i:$port 2>/dev/null) && [ -n "$pid" ] && kill $pid 2>/dev/null || true
done

sleep 1

# Start all parameter servers
i=0
while IFS= read -r addr; do
  PORT=$(echo "$addr" | cut -d: -f2)
  setsid gnome-terminal --title="server-$i" -- bash -c "
    cd $ROOT
    PORT=$PORT RUST_LOG=info cargo run --release -p parameter_server
    exec bash
  " &
  i=$((i + 1))
  sleep 0.5
done < <(python3 -c "
import json
d = json.load(open('$ORCHESTUI_DIR/training.json'))
print('\n'.join(d['algorithm']['parameter_server']['server_addrs']))
")

sleep 1

# Start all workers
i=0
while IFS= read -r addr; do
  PORT=$(echo "$addr" | cut -d: -f2)
  setsid gnome-terminal --title="worker-$i" -- bash -c "
    cd $ROOT
    PORT=$PORT RUST_LOG=info cargo run --release -p worker
    exec bash
  " &
  i=$((i + 1))
  sleep 0.5
done < <(python3 -c "
import json
d = json.load(open('$ORCHESTUI_DIR/training.json'))
print('\n'.join(d['worker_addrs']))
")

sleep 1

# Orchestrator log tail
setsid gnome-terminal --title="orchestrator-log" -- bash -c "
  touch /tmp/ono-orchestrator.log
  tail -f /tmp/ono-orchestrator.log
  exec bash
" &

sleep 0.5

# Run TUI in current terminal
cd $ROOT
RUST_LOG=info cargo run --release -p orchestui 2>/tmp/ono-orchestrator.log