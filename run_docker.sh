#!/usr/bin/bash

set -e
trap 'echo "Running Cleanup..."; rm -f "$OUTPUT_PATH"' EXIT

SCRIPTS_DIR="docker"
export CONFIG_PATH="$SCRIPTS_DIR/config.json"
export OUTPUT_PATH="compose.yml"

echo "Generating compose file..."
./$SCRIPTS_DIR/gen_compose.py

echo "Executing docker compose..."
# Acá hay que poner lo que ejecuta todo, algo tipo `docker compose up`.

