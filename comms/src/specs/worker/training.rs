use std::num::NonZeroUsize;

use serde::{Deserialize, Serialize};

use crate::specs::server::OptimizerSpec;

use super::algorithm::AlgorithmSpec;

/// Training configuration for a worker instance.
#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingSpec {
    pub algorithm: AlgorithmSpec,
    pub optimizer: OptimizerSpec,
    pub offline_steps: usize,
    pub epochs: NonZeroUsize,
}
