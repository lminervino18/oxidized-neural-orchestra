# Oxidized Neural Orchestra

A modular, distributed Rust framework for training and inference of neural networks across multiple machines. Designed for reproducible experiments comparing synchronization strategies and communication patterns.

---

## Table of Contents

- [Overview](#overview)
- [Goals](#goals)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [How it Works](#how-it-works)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Contributing & Project Info](#contributing--project-info)

---

## Overview

Oxidized Neural Orchestra aims to provide a research-ready platform that scales deep learning workloads horizontally, enabling controlled comparisons of synchronization methods (All‑Reduce, Parameter Server, hybrids) and their effect on convergence and time-to-train.

## Goals

- Support distributed training and inference from single-node to clusters.
- Enable data-parallel experiments with reproducible setups.
- Provide interchangeable synchronization strategies to compare behavior under realistic network conditions.
- Offer tooling for experiments, metrics, and Python interoperability.

## Key Features

- Pluggable synchronization implementations:
  - All‑Reduce (decentralized, synchronous)
  - Parameter Server (centralized, potentially asynchronous)
  - Hybrid/adaptive strategies
- Actor-based orchestration for clear coordination and failure boundaries.
- Simple TCP communication layer (framing pluggable).
- Research-oriented instrumentation for throughput, latency, and convergence metrics.

## Architecture

- Orchestrator coordinates rounds/epochs, assigns tasks, and aggregates results.
- Workers execute forward/backward passes on sharded data and participate in the selected sync protocol.
- Communication is over TCP with an explicit read/write split to avoid lock contention; planned async refactor will replace blocking threads with Tokio-based framed streams.

Example component layout:
- `src/orchestrator.rs` — Orchestrator actor and per-connection worker handler
- `src/worker.rs` — Receiver / Producer / Sender actor pipeline
- `src/communication.rs` — TCP helpers with read/write split
- `src/main.rs` — Entrypoint for orchestrator or worker

## How it Works

1. Orchestrator initiates a round (Kickoff).
2. Tasks are dispatched to all connected workers.
3. Workers process tasks (local forward/backward, gradient exchange depending on protocol).
4. Results are collected; Orchestrator emits an AllDone event and advances the loop.
5. Repeat for subsequent rounds/epochs.

The current implementation uses blocking reads in separate threads per connection to simplify full‑duplex behavior. An async refactor (Tokio + framed streams) is planned to scale to many connections without per-connection threads.

## Usage

Quick start (conceptual)
```bash
# Run orchestrator
cargo run --bin orchestrator

# Run a worker
cargo run --bin worker -- --connect 127.0.0.1:9000
```
Refer to the code in `src/` for configuration flags and runtime options.

## Evaluation

Intended evaluation targets:
- Convergence speed across synchronization strategies.
- Throughput and wall-clock time-to-train under varying worker counts.
- Robustness under network variability (latency, packet loss).
- Scalability: current blocking I/O supports small-to-medium clusters; async refactor will enable large-scale experiments.

## Contributing & Project Info

Faculty of Engineering, University of Buenos Aires — Final Degree Project.

Contributions and issues welcome. Suggested next steps:
- Add framing (line or length-delimited) and tests.
- Migrate I/O to async (Tokio) with framed streams/sinks.
- Add heartbeats, reconnection policies, and richer metrics.

License: (add license file / specify license here)