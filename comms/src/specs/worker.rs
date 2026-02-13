use std::net::SocketAddr;

use serde::{Deserialize, Serialize};

use super::machine_learning::TrainerSpec;

/// Wire-level bootstrap specification for a worker instance.
///
/// This type is exchanged over the network during worker bootstrap.
#[derive(Debug, Serialize, Deserialize)]
pub struct WorkerSpec<'a> {
    pub worker_id: usize,
    #[serde(borrow)]
    pub trainer: TrainerSpec<'a>,
    pub algorithm: AlgorithmSpec,
}

/// Distributed training algorithm selection.
///
/// Only `parameter_server` is currently implemented.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlgorithmSpec {
    ParameterServer { server_ip: SocketAddr },
}
