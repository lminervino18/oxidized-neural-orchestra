use std::net::IpAddr;

use serde::{Deserialize, Serialize};

pub mod algorithm;
pub mod model;
pub mod training;

use super::machine_learning::TrainerSpec;
pub use algorithm::AlgorithmSpec;
pub use model::{ActFnSpec, LayerSpec, ModelSpec};
pub use training::TrainingSpec;

/// Wire-level bootstrap specification for a worker instance.
#[derive(Debug, Serialize, Deserialize)]
pub struct WorkerSpec {
    pub worker_id: usize,
    pub trainer: TrainerSpec,
    pub server_addr: IpAddr,
}
