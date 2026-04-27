use serde::{Deserialize, Serialize};

use super::machine_learning::{DatasetSpec, TrainerSpec};
use crate::sparse::Float01;

/// Distributed training algorithm selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlgorithmSpec {
    ParameterServer {
        server_addrs: Vec<String>,
        server_sizes: Vec<usize>,
        server_ordering: Vec<usize>,
        #[serde(default)]
        server_session_ids: Vec<u64>,
    },
    AllReduce {
        worker_addrs: Vec<String>,
    },
}

/// Message serializer for gradient sparse compression.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SerializerSpec {
    Base,
    SparseCapable { r: Float01, seed: Option<u64> },
}

/// The specification for the `Worker`.
#[derive(Debug, Serialize, Deserialize)]
pub struct WorkerSpec {
    pub worker_id: usize,
    pub trainer: TrainerSpec,
    pub dataset: DatasetSpec,
    pub algorithm: AlgorithmSpec,
    pub serializer: SerializerSpec,
}
