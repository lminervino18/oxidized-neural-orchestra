# 0. Define the arguments' default values.
ARG PKG=orchestrator
ARG MODE

# 1. Install cargo-chef.
FROM rust:1.92-slim AS chef
RUN cargo install cargo-chef
WORKDIR /app

# 2. Prepare this package's dependencies.
FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

# 3. Cook the dependencies' recipe and build the package.
FROM chef AS builder
COPY --from=planner /app/recipe.json recipe.json
ARG MODE
ARG PKG

RUN if [ "$MODE" = "release" ]; then export FLAG="--release"; fi && \
  cargo chef cook $FLAG --recipe-path recipe.json

COPY Cargo.* .
COPY $PKG    ./$PKG
COPY comms   ./comms

RUN if [ "$MODE" = "release" ]; then export FLAG="--release"; fi && \
  cargo build $FLAG --bin $PKG

RUN mv target/$MODE/$PKG /app/binary

# 4. Execute the package.
FROM debian:bookworm-slim
RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/binary /usr/local/bin
ENTRYPOINT ["/usr/local/bin/binary"]