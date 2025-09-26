#!/bin/bash

# Address and port for the Orchestrator and Workers
ADDRESS="127.0.0.1:6700"

# Start the Orchestrator in the background
echo "Starting Orchestrator..."
gnome-terminal -- bash -c "cargo run orchestrator $ADDRESS 4; exec bash" &  # Start Orchestrator in a new terminal
sleep 2  # Wait for Orchestrator to start and be ready


echo "Orchestrator is now listening. Starting Workers..."

# Start 4 Workers in new terminals
for i in {1..4}
do
    gnome-terminal -- bash -c "cargo run worker $ADDRESS; exec bash" &  # Start Worker in a new terminal
    echo "Worker $i started"
    sleep 1  # Wait 1 second between starting each Worker
done

echo "All Workers and Orchestrator have been started in separate terminals."
f