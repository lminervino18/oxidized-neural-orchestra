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

/// The specification for the `Model` trait.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelSpec {
    Sequential { layers: Vec<LayerSpec> },
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
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
// sería bueno mandar `chunk_size` como parte del `DatasetSpec`?
pub struct DatasetSpec {
    pub size: usize,
    pub x_size: usize,
    pub y_size: usize,
}

/// The specification for the `LossFn` enum.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LossFnSpec {
    Mse,
}

/// The specification for the `Trainer` struct.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainerSpec {
    pub model: ModelSpec,
    pub optimizer: OptimizerSpec,
    pub dataset: DatasetSpec,
    pub loss_fn: LossFnSpec,
    pub batch_size: NonZeroUsize,
    pub offline_epochs: usize,
    pub seed: Option<u64>,
}
