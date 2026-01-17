# 0. Define the package's name.
ARG PKG=parameter_server

# 1. Install cargo-chef.
FROM rust:1.92 AS chef
RUN cargo install cargo-chef
WORKDIR /app

# 2. Prepare this package's dependencies.
FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

# 3. Cook the dependencies' recipe and build the package.
FROM chef AS builder
COPY --from=planner /app/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json
ARG PKG

  # Base dependencies.
  COPY Cargo.* .
  COPY $PKG    ./$PKG

  # Extra dependencies.
  COPY comms ./comms

RUN cargo build --release --bin $PKG
RUN mv target/release/$PKG /app/binary

# 4. Execute the package.
FROM debian:bookworm-slim
COPY --from=builder /app/binary /usr/local/bin
ENTRYPOINT /usr/local/bin/binary
