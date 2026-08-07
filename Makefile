# Oxidized Neural Orchestra — task runner.
# Run `make help` to list every target. Override variables inline, e.g.
#   make node PORT=40001 RELEASE=0
#   make bench ARGS="--suite convergence --model lenet5"

# ── Config ────────────────────────────────────────────────
RELEASE ?= 1
PORT    ?= 40000
NODES   ?= 5
SERVERS ?= 2
ARGS    ?=
PYTHON  ?= .venv/bin/python
CARGO_RELEASE := $(if $(filter 1,$(RELEASE)),--release,)
DOCKER_RELEASE := $(if $(filter 1,$(RELEASE)),--release,)

.DEFAULT_GOAL := help

# ── Help ──────────────────────────────────────────────────
help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-20s\033[0m %s\n",$$1,$$2}'

# ── Setup ─────────────────────────────────────────────────
venv: ## Create the venv + the build tool (maturin)
	@test -d .venv || python3 -m venv .venv
	.venv/bin/pip install -q --upgrade pip maturin

install: venv ## Runtime install: the orchestra module only (no benchmark deps)
	.venv/bin/pip install -e ./orchestra-py
	cargo fetch

install-dev: venv ## Dev install: orchestra + benchmark & lint deps (torch, numpy, ruff)
	.venv/bin/pip install -e "./orchestra-py[bench,dev]"
	cargo fetch

dataset: ## Download MNIST into datasets/
	$(PYTHON) scripts/download_mnist.py

# ── Build ─────────────────────────────────────────────────
build: ## Build the whole Rust workspace
	cargo build $(CARGO_RELEASE)

py-build: ## Compile the orchestra-py extension into the venv
	cd orchestra-py && ../.venv/bin/maturin develop $(CARGO_RELEASE)

# ── Run: components ───────────────────────────────────────
node: ## Run one node (PORT=40000)
	PORT=$(PORT) RUST_LOG=info cargo run -p node $(CARGO_RELEASE)

orchestui: ## Launch the TUI (loads model.json / training.json)
	cargo run -p orchestui $(CARGO_RELEASE)

orchestui-cluster: ## Bring up docker nodes + the TUI (end-to-end)
	bash orchestui/run.sh $(DOCKER_RELEASE)

orchestra-py: ## Run the Python driver (needs a running cluster)
	NODES=$(NODES) SERVERS=$(SERVERS) $(PYTHON) orchestra-py/local.py

# ── Docker cluster ────────────────────────────────────────
cluster-up: ## Start N docker nodes (NODES=5)
	python3 docker/compose_up.py --nodes $(NODES) $(DOCKER_RELEASE)

cluster-down: ## Tear down the docker cluster
	docker compose -f compose.yaml down --remove-orphans

# ── Benchmarks ────────────────────────────────────────────
bench: ## Run benchmarks (ARGS="--suite convergence")
	$(PYTHON) benchmarks/run_issue_benchmarks.py $(ARGS)

bench-plots: ## Rebuild benchmark plots/README from history
	$(PYTHON) benchmarks/run_issue_benchmarks.py --plots-only

# ── Quality ───────────────────────────────────────────────
test: ## Run the Rust test suite
	cargo test --workspace

fmt: ## Format Rust code
	cargo fmt --all

fmt-check: ## Check Rust formatting without writing
	cargo fmt --all --check

lint: ## Clippy over the whole workspace
	cargo clippy --workspace --all-targets

py-lint: ## Lint Python with ruff
	.venv/bin/ruff check .

py-fmt: ## Format Python with ruff
	.venv/bin/ruff format .

check: fmt-check lint test ## CI-style: Rust format check + clippy + tests

# ── Clean ─────────────────────────────────────────────────
clean: ## Remove build artifacts and temp outputs
	cargo clean
	rm -f compose.yaml *.log
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true

.PHONY: help venv install install-dev dataset build py-build node orchestui \
        orchestui-cluster orchestra-py cluster-up cluster-down \
        bench bench-plots test fmt fmt-check lint py-lint py-fmt check clean
