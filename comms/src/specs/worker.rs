use std::num::NonZeroUsize;

use serde::{Deserialize, Serialize};

use super::strategy::StrategySpec;

/// The specification for the worker execution policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionSpec {
    Default,
    Batching {
        microbatch_size: NonZeroUsize,
        grad_accum_steps: NonZeroUsize,
    },
}

impl Default for ExecutionSpec {
    fn default() -> Self {
        Self::Default
    }
}

/// The specification for an external artifact needed by a worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactSpec {
    pub uri: String,
    pub sha256: Option<String>,
}

/// The specification for worker artifacts.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WorkerArtifacts {
    pub model: Option<ArtifactSpec>,
    pub dataset: Option<ArtifactSpec>,
}

/// The specification for the worker bootstrap.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerSpec {
    pub worker_id: usize,
    pub steps: NonZeroUsize,
    pub num_params: NonZeroUsize,
    pub strategy: StrategySpec,
    #[serde(default)]
    pub artifacts: WorkerArtifacts,
    #[serde(default)]
    pub execution: ExecutionSpec,
    pub seed: Option<u64>,
}
