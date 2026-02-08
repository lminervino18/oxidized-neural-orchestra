mod algorithms;
mod configs;
mod session;

use std::io;

use configs::to_specs_adapter;
use session::Session;

use crate::configs::{ModelConfig, TrainingConfig};

pub fn train(model: ModelConfig, training: TrainingConfig) -> io::Result<Session> {
    let (worker_spec, server_spec) = to_specs_adapter(model, training)?;

    if let Some(server) = server_spec {
        // agregar a la sesion el spec del server
    }

    // agregar a la sesion el spec del worker

    Ok(Session {})
}
