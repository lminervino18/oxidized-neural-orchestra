use std::num::NonZeroUsize;

use serde::{Deserialize, Serialize};

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
