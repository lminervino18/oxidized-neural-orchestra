#!/usr/bin/bash

set -e

SCRIPTS_DIR="docker"
export CONFIG_PATH="$SCRIPTS_DIR/config.json"
export OUTPUT_PATH="compose.yml"

echo "Generating compose file..."
./$SCRIPTS_DIR/gen_compose.py

echo "Executing docker compose..."
docker compose -f "$OUTPUT_PATH" up --build -d --remove-orphans
