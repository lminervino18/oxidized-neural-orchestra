use serde::{Deserialize, Serialize};

use super::machine_learning::{
    ParamGenSpec, {DatasetSpec, TrainerSpec},
};
use crate::sparse::Float01;

/// Distributed training algorithm selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlgorithmSpec {
    ParameterServer {
        server_addrs: Vec<String>,
        server_sizes: Vec<usize>,
        server_ordering: Vec<usize>,
    },
    AllReduce {
        worker_addrs: Vec<String>,
        param_gen: ParamGenSpec,
    },
}

/// Message serializer for gradient sparse compression.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SerializerSpec {
    Base,
    SparseCapable { r: Float01 },
}

/// The specification for the `Worker`.
#[derive(Debug, Serialize, Deserialize)]
pub struct WorkerSpec {
    pub worker_id: usize,
    pub trainer: TrainerSpec,
    pub dataset: DatasetSpec,
    pub algorithm: AlgorithmSpec,
    pub serializer: SerializerSpec,
    pub seed: Option<u64>,
}
