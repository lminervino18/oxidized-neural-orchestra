use std::num::NonZeroUsize;

use serde::{Deserialize, Serialize};

pub mod algorithm;
pub mod model;
pub mod training;

use super::machine_learning::TrainerSpec;
pub use algorithm::AlgorithmSpec;
pub use model::{ActFnSpec, LayerSpec, ModelSpec};
pub use training::TrainingSpec;

/// Wire-level bootstrap specification for a worker instance.
#[derive(Debug, Serialize, Deserialize)]
pub struct WorkerSpec {
    /// Worker identifier assigned by the orchestrator.
    pub worker_id: usize,
    /// Number of steps to execute.
    pub steps: NonZeroUsize,
    /// Expected parameter count for `weights` and `gradient` payloads.
    pub trainer: TrainerSpec,
}
