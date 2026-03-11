pub mod configs;
mod error;
mod session;

use std::net::ToSocketAddrs;

use configs::Adapter;
use error::{OrchErr, Result};
pub use session::Session;
pub use session::TrainingEvent;

use crate::configs::{ModelConfig, TrainingConfig, Validator};

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
/// Returns an `OrchErr` if config validation fails or connecting to any worker or server fails.
pub fn train<A: ToSocketAddrs>(model: ModelConfig, training: TrainingConfig<A>) -> Result<Session> {
    let validator = Validator::new();
    validator.validate(&model, &training)?;

    let adapter = Adapter::new();
    let (workers, partitions, servers) = adapter.adapt_configs(model, &training)?;

    Session::new(workers, partitions, servers)
}
