pub mod configs;
mod session;

use std::{io, net::ToSocketAddrs};

use configs::Adapter;
use session::Session;

pub use session::TrainingEvent;

use crate::configs::{ModelConfig, TrainingConfig};

/// Starts the distributed training process and returns an active session.
///
/// # Errors
/// Returns an `io::Error` if connecting to any worker or server fails.
pub fn train<A: ToSocketAddrs>(
    model: ModelConfig,
    training: TrainingConfig<A>,
) -> io::Result<Session> {
    log::info!("adapting configs");
    let adapter = Adapter::new();
    let (workers, servers) = adapter.adapt_configs(model, training)?;
    log::info!(
        "connecting to {} worker(s) and {} server(s)",
        workers.len(),
        servers.len()
    );
    Session::new(workers, servers)
}
