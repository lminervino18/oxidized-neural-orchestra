#!/bin/bash

set -e

MODULE="${1:-orchestrator}"
NETWORK_NAME="distributed-training_training-network"
CONFIG_PATH="docker/config.json"

MODE=$(python3 -c "import json; c=json.load(open('$CONFIG_PATH')); print('release' if c['release'] else 'debug')")
WORKERS=$(python3 -c "import json; c=json.load(open('$CONFIG_PATH')); print(c['workers'])")
SERVERS=$(python3 -c "import json; c=json.load(open('$CONFIG_PATH')); print(c['servers'])")

case "$MODULE" in
    orchestrator)
        DOCKERFILE="orchestrator/Dockerfile"
        IMAGE_NAME="orchestrator"
        ;;
    orchestra-py)
        DOCKERFILE="orchestra-py/Dockerfile"
        IMAGE_NAME="orchestra-py"
        ;;
    *)
        echo "Unknown module: $MODULE"
        echo "Usage: $0 [orchestrator|orchestra-py]"
        exit 1
        ;;
esac

echo "Building $IMAGE_NAME image ($MODE)..."

docker build               \
    --build-arg MODE=$MODE \
    -t $IMAGE_NAME         \
    -f $DOCKERFILE .

echo "Running $IMAGE_NAME container..."

docker run --rm                       \
    --name $IMAGE_NAME                \
    --network $NETWORK_NAME           \
    -e WORKERS="$WORKERS"             \
    -e SERVERS="$SERVERS"             \
    -e DATASET_PATH="/dataset"        \
    -v "$(pwd)/data/dataset:/dataset:ro"   \
    $IMAGE_NAME