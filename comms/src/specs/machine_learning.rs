use std::num::NonZeroUsize;

use serde::{Deserialize, Serialize};

/// The specification for the `ActFn` enum.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActFnSpec {
    Sigmoid { amp: f32 },
}

/// The specification for the `Layer` enum.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerSpec {
    Dense {
        dim: (usize, usize),
        act_fn: Option<ActFnSpec>,
    },
}

/// The specification for the `Optimizer` trait.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OptimizerSpec {
    Adam {
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
    },
    GradientDescent {
        learning_rate: f32,
    },
    GradientDescentWithMomentum {
        learning_rate: f32,
        momentum: f32,
    },
}

/// The specification for the `Dataset`.
#[derive(PartialEq, Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct DatasetSpec {
    /// The size of all the dataset samples in bytes.
    pub x_size_bytes: u64,
    /// The size of all the dataset labels in bytes.
    pub y_size_bytes: u64,
    // TODO: no sé qué nombres poner...
    /// The length of each sample.
    pub x_size: NonZeroUsize,
    /// The length of each label.
    pub y_size: NonZeroUsize,
}

/// The specification for the `LossFn` enum.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LossFnSpec {
    Mse,
    CrossEntropy,
}

/// The specification for the `Trainer` struct.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainerSpec {
    pub layers: Vec<LayerSpec>,
    pub optimizer: OptimizerSpec,
    pub loss_fn: LossFnSpec,
    pub offline_epochs: usize,
    pub max_epochs: NonZeroUsize,
    pub batch_size: NonZeroUsize,
    pub seed: Option<u64>,
}
