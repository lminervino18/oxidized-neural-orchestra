use std::net::SocketAddr;

use serde::{Deserialize, Serialize};

use super::machine_learning::TrainerSpec;

/// Distributed training algorithm selection.
///
/// Only `parameter_server` is currently implemented.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlgorithmSpec {
    ParameterServer {
        server_addrs: Vec<SocketAddr>,
        server_sizes: Vec<usize>,
        server_ordering: Vec<usize>,
    },
}

/// Wire-level bootstrap specification for a worker instance.
///
/// This type is exchanged over the network during worker bootstrap.
#[derive(Debug, Serialize, Deserialize)]
pub struct WorkerSpec {
    pub worker_id: usize,
    pub trainer: TrainerSpec,
    pub algorithm: AlgorithmSpec,
}
