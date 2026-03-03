pub mod configs;
pub mod error;
mod session;

use configs::Adapter;
use session::Session;

pub use error::OrchestratorError;
pub use session::TrainingEvent;

use crate::configs::{ModelConfig, TrainingConfig};

/// Starts the distributed training process and returns an active session.
///
/// # Args
/// * `model` - The model architecture configuration.
/// * `training` - The training configuration, including worker and server addresses.
///
/// # Returns
/// A new ongoing session.
///
/// # Errors
/// Returns an `OrchestratorError` if config validation fails or connecting to
/// any worker or server fails.
pub fn train(model: ModelConfig, training: TrainingConfig) -> Result<Session, OrchestratorError> {
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
