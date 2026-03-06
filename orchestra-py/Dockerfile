# 0. Define build arguments.
ARG MODE=debug

# 1. Build Rust dependencies with cargo-chef.
FROM rust:1.92-slim AS chef
RUN cargo install cargo-chef
WORKDIR /app

# 2. Prepare the dependency recipe.
FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

# 3. Cook dependencies and build the Python extension.
FROM chef AS builder
COPY --from=planner /app/recipe.json recipe.json
ARG MODE

RUN if [ "$MODE" = "release" ]; then export FLAG="--release"; fi && \
    cargo chef cook $FLAG --recipe-path recipe.json

COPY Cargo.* .
COPY comms          ./comms
COPY orchestrator   ./orchestrator
COPY orchestra-py   ./orchestra-py

RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*

RUN pip install maturin --break-system-packages || \
    (apt-get update && apt-get install -y python3-pip && pip install maturin --break-system-packages)

RUN cd orchestra-py && \
    if [ "$MODE" = "release" ]; then maturin build --release; else maturin build; fi && \
    pip install /app/target/wheels/*.whl --break-system-packages

# 4. Runtime — stay in builder to avoid Python version mismatch.
FROM builder

COPY orchestra-py/train.py /app/train.py
WORKDIR /app
ENTRYPOINT ["python3", "train.py"]

