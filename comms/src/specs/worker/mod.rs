use std::num::NonZeroUsize;

use serde::{Deserialize, Serialize};

/// Wire-level bootstrap specification for a worker instance.
#[derive(Debug, Serialize, Deserialize)]
pub struct WorkerSpec {
    /// Worker identifier assigned by the orchestrator.
    pub worker_id: usize,
    /// Number of steps to execute.
    pub steps: NonZeroUsize,
    /// Expected parameter count for `weights` and `gradient` payloads.
    pub num_params: NonZeroUsize,
    /// Model selection and configuration.
    pub model: ModelSpec,
    /// Training configuration.
    pub training: TrainingSpec,
    /// Optional seed for deterministic initialization.
    pub seed: Option<u64>,
}

/// Model selection and configuration.
///
/// The initial variants exist to keep the system testable while the model
/// runtime is expanded incrementally.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelSpec {
    /// No-op model used for protocol tests.
    Noop,
    /// Mock model used for end-to-end tests (gradient = 2 * weights).
    Mock,
    /// Feed-forward network (placeholder for the next refactor steps).
    FeedForward { layers: Vec<LayerSpec> },
}

/// Feed-forward layer specification.
#[derive(Debug, Serialize, Deserialize)]
pub struct LayerSpec {
    pub input: NonZeroUsize,
    pub output: NonZeroUsize,
    pub act_func: ActFuncSpec,
}

/// Activation function specification.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActFuncSpec {
    Sigmoid { amp: f32 },
}

/// Distributed training algorithm selection.
///
/// Only `parameter_server` is currently implemented.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlgorithmSpec {
    ParameterServer { server_ips: Vec<String> },
    AllReduce { peers: Vec<String> },
    StrategySwitch,
}

/// Training configuration for a worker instance.
#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingSpec {
    pub algorithm: AlgorithmSpec,
    pub optimizer: crate::specs::server::OptimizerSpec,
    pub offline_steps: usize,
    pub epochs: NonZeroUsize,
}
