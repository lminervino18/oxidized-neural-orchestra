#!/bin/bash

set -e

IMAGE_NAME="orchestrator"
NETWORK_NAME="distributed-training_training-network"
MODE="debug" # or "release"

echo "Building Orchestrator Image ($MODE)..."

docker build               \
    --build-arg MODE=$MODE \
    -t $IMAGE_NAME         \
    -f orchestrator/Dockerfile .

echo "Running Orchestrator Container..."

docker run --rm             \
    --name orchestrator     \
    --network $NETWORK_NAME \
    -e RUST_LOG=$MODE \
    $IMAGE_NAME
