pub mod configs;
mod session;

use std::{io, net::ToSocketAddrs};

use configs::Adapter;
use session::Session;

use crate::configs::{ModelConfig, TrainingConfig};

/// Initiazes the distributed training process.
///
/// # Arguments
/// * `model` - The model's configuration.
/// * `training` - The training's configuration.
///
/// # Returns
/// A new ongoing session or an io error if occurred.
pub fn train<A: ToSocketAddrs>(
    model: ModelConfig,
    training: TrainingConfig<A>,
) -> io::Result<Session> {
    let adapter = Adapter::new();
    let (workers, servers) = adapter.adapt_configs(model, training)?;

    println!("workers: {workers:#?}");
    println!("servers: {servers:#?}");

    Session::new(workers, servers)
}
