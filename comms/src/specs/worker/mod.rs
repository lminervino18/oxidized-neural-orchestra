use std::num::NonZeroUsize;

use serde::{Deserialize, Serialize};

/// The wire-level bootstrap specification for a worker instance.
///
/// This type is intentionally minimal: it contains only what the infrastructure
/// needs to instantiate and run a worker process.
#[derive(Debug, Serialize, Deserialize)]
pub struct WorkerSpec {
    pub worker_id: usize,
    pub steps: NonZeroUsize,
    pub num_params: NonZeroUsize,
    pub strategy: StrategySpec,
    pub seed: Option<u64>,
}

/// The specification for selecting a training strategy.
///
/// This is an explicit, typed selector. It avoids stringly-typed "kind" fields
/// and removes opaque parameter blobs from the wire protocol.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StrategySpec {
    Noop,
    Mock,
}
