use std::num::NonZeroUsize;

use serde::{Deserialize, Serialize};

use crate::floats::{Float01, FloatPositive};

/// The specification for the `Distribution` trait.
#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DistributionSpec {
    Uniform { low: f32, high: f32 },
    UniformInclusive { low: f32, high: f32 },
    XavierUniform { fan_in: usize, fan_out: usize },
    LecunUniform { fan_in: usize },
    Normal { mean: f32, std_dev: f32 },
    Kaiming { fan_in: usize },
    Xavier { fan_in: usize, fan_out: usize },
    Lecun { fan_in: usize },
}

/// The specification for the `ParamGen` trait.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParamGenSpec {
    Const {
        value: f32,
        limit: usize,
    },
    Inline {
        params: Vec<f32>,
    },
    Rand {
        distribution: DistributionSpec,
        limit: usize,
    },
    Chained {
        specs: Vec<ParamGenSpec>,
    },
}

impl ParamGenSpec {
    pub fn size(&self) -> usize {
        match self {
            ParamGenSpec::Const { limit, .. } => *limit,
            ParamGenSpec::Rand { limit, .. } => *limit,
            ParamGenSpec::Inline { params } => params.len(),
            ParamGenSpec::Chained { specs } => specs.iter().map(|spec| spec.size()).sum(),
        }
    }
}

/// The specification for the `ActFn` enum.
#[derive(Debug, PartialEq, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActFnSpec {
    Sigmoid { amp: f32 },
    Tanh { amp: f32 },
    ReLU,
    Softmax,
}

/// The specification for the `Layer` enum.
#[derive(Debug, PartialEq, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerSpec {
    Dense {
        dim: (usize, usize),
        act_fn: Option<ActFnSpec>,
    },
    Conv {
        input_dim: (usize, usize, usize),
        kernel_dim: (usize, usize, usize),
        stride: usize,
        padding: usize,
        act_fn: Option<ActFnSpec>,
    },
}

/// The specification for the `Optimizer` trait.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OptimizerSpec {
    Adam {
        learning_rate: FloatPositive,
        beta1: Float01,
        beta2: Float01,
        epsilon: FloatPositive,
    },
    GradientDescent {
        learning_rate: FloatPositive,
    },
    GradientDescentWithMomentum {
        learning_rate: FloatPositive,
        momentum: Float01,
    },
}

/// The specification for the `Dataset`.
#[derive(PartialEq, Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct DatasetSpec {
    pub x_size: NonZeroUsize,
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
    pub dataset: DatasetSpec,
    pub loss_fn: LossFnSpec,
    pub offline_epochs: usize,
    pub max_epochs: NonZeroUsize,
    pub batch_size: NonZeroUsize,
    pub seed: Option<u64>,
}
