use std::num::NonZeroUsize;

use serde::{Deserialize, Serialize};

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
            ParamGenSpec::Chained { specs } => specs.iter().map(|spec| spec.size()).sum(),
        }
    }
}

/// The specification for the `ActFn` enum.
#[derive(Debug, PartialEq, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActFnSpec {
    Sigmoid { amp: f32 },
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
        /// The in channels, height and width of the input.
        input_dim: (usize, usize, usize),
        /// The filters, in channels, and size of the square kernel.
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
