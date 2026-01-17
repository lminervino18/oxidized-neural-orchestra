#!/usr/bin/bash

SCRIPTS_DIR="docker"
CONFIG_PATH="$SCRIPTS_DIR/config.json"
OUTPUT_PATH="compose.yml"

CONFIG_PATH=$CONFIG_PATH OUTPUT_PATH=$OUTPUT_PATH ./$SCRIPTS_DIR/gen_compose.py

# Acá hay que poner lo que ejecuta todo, algo tipo `docker compose up`
echo "Executing training..."

rm $OUTPUT_PATH
