use serde::{Deserialize, Serialize};

use super::{
    algorithm::AlgorithmSpec,
    machine_learning::{DatasetSpec, TrainerSpec},
};

/// The specification for the `Worker`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkerSpec {
    pub worker_id: usize,
    pub trainer: TrainerSpec,
    pub dataset: DatasetSpec,
    pub algorithm: AlgorithmSpec,
}