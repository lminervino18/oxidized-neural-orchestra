mod app;
mod config;
mod naming;
mod ui;

use std::{
    env,
    fs::File,
    io::{self, IsTerminal},
    sync::Mutex,
};

use anyhow::Result;
use tracing_subscriber::{fmt, EnvFilter};

const LOG_FILE: &str = "ono-orchestui.log";

/// Initializes logging, defaulting to `info` when `RUST_LOG` is unset.
///
/// Logs go to `LOG_FILE` in the temp directory when stderr is a terminal, since the TUI renders
/// there and inline log lines would corrupt the frame. Otherwise they go to stderr.
fn init_logging() {
    let builder = fmt().with_env_filter(
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
    );

    let log_file = io::stderr()
        .is_terminal()
        .then(|| File::create(env::temp_dir().join(LOG_FILE)).ok())
        .flatten();

    match log_file {
        Some(file) => builder
            .with_ansi(false)
            .with_writer(Mutex::new(file))
            .init(),
        None => builder.with_writer(io::stderr).init(),
    }
}

fn main() -> Result<()> {
    init_logging();
    app::run::run()
}
