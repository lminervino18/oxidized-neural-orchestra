pub mod configs;
pub mod error;
mod session;

use std::net::ToSocketAddrs;

use configs::Adapter;
use session::Session;

pub use error::OrchestratorError;
pub use session::TrainingEvent;

use crate::configs::{ModelConfig, TrainingConfig};

/// Starts the distributed training process and returns an active session.
///
/// # Errors
/// Returns an `OrchestratorError` if connecting to any worker or server fails.
pub fn train<A: ToSocketAddrs>(
    model: ModelConfig,
    training: TrainingConfig<A>,
) -> Result<Session, OrchestratorError> {
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
