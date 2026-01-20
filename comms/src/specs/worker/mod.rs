use std::num::NonZeroUsize;

use serde::{Deserialize, Serialize};

/// The wire-level bootstrap specification for a worker instance.
#[derive(Debug, Serialize, Deserialize)]
pub struct WorkerSpec {
    pub worker_id: usize,
    pub steps: NonZeroUsize,
    pub num_params: NonZeroUsize,
    pub strategy: StrategySpec,
    pub seed: Option<u64>,
}

/// The specification for selecting a training strategy.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StrategySpec {
    Noop,
    Mock,
}
