mod app;
mod config;
mod naming;
mod ui;

use anyhow::Result;
use tracing_subscriber::{fmt, EnvFilter};

/// Initializes stderr logging, defaulting to `info` when `RUST_LOG` is unset.
fn init_logging() {
    fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();
}

fn main() -> Result<()> {
    init_logging();
    app::run::run()
}
