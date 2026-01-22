use std::num::NonZeroUsize;

use serde::{Deserialize, Serialize};

/// Wire-level bootstrap specification for a worker instance.
#[derive(Debug, Serialize, Deserialize)]
pub struct WorkerSpec {
    /// Worker identifier assigned by the orchestrator.
    pub worker_id: usize,
    /// Number of training steps to execute.
    pub steps: NonZeroUsize,
    /// Expected parameter count for `weights` and `gradient` payloads.
    pub num_params: NonZeroUsize,
    /// Training strategy selection.
    pub strategy: StrategySpec,
    /// Optional seed for strategy/model initialization.
    pub seed: Option<u64>,
}

/// Training strategy selector.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StrategySpec {
    Noop,
    Mock,
}
