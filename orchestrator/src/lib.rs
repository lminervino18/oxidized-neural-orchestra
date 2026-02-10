mod algorithms;
mod configs;
mod session;

use std::io;

use configs::to_specs_adapter;
use session::Session;

use crate::configs::{ModelConfig, TrainingConfig};

pub fn train(model: ModelConfig, training: TrainingConfig) -> io::Result<Session> {
    let (worker_addrs, worker_spec, server) = to_specs_adapter(model, training)?;

    if let Some((server_addr, server_spec)) = server {
        return Session::new(worker_addrs, worker_spec, server_addr, server_spec);
    }

    unimplemented!("AllReduce is not yet implemented, a server must be specified");
}
