use serde::{Deserialize, Serialize};

pub mod algorithm;
pub mod model;

use super::machine_learning::TrainerSpec;
pub use algorithm::AlgorithmSpec;

/// Wire-level bootstrap specification for a worker instance.
#[derive(Debug, Serialize, Deserialize)]
pub struct WorkerSpec {
    pub worker_id: usize,
    pub trainer: TrainerSpec,
    pub algorithm: AlgorithmSpec,
}
