use serde::{Deserialize, Serialize};

use super::{
    algorithm::AlgorithmSpec,
    machine_learning::{DatasetSpec, TrainerSpec},
};
use crate::sparse::Float01;

/// Message serializer for gradient sparse compression.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SerializerSpec {
    Base,
    SparseCapable { r: Float01, seed: Option<u64> },
}

/// The specification for the `Worker`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkerSpec {
    pub worker_id: usize,
    pub trainer: TrainerSpec,
    pub dataset: DatasetSpec,
    pub algorithm: AlgorithmSpec,
    pub serializer: SerializerSpec,
}
