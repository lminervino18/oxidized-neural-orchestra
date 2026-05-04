use std::num::NonZeroUsize;

use serde::{Deserialize, Serialize};

/// The `ParamGen` configuration.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParamGenConfig {
    Const { value: f32 },
    Uniform { low: f32, high: f32 },
    UniformInclusive { low: f32, high: f32 },
    XavierUniform,
    LecunUniform,
    Normal { mean: f32, std_dev: f32 },
    Kaiming,
    Xavier,
    Lecun,
}

/// The `ActFn` configuration.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActFnConfig {
    Sigmoid { amp: f32 },
    Softmax,
}

/// The `Layer` configuration.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerConfig {
    Dense {
        output_size: NonZeroUsize,
        init: ParamGenConfig,
        #[serde(default)]
        act_fn: Option<ActFnConfig>,
    },
    Conv {
        /// The in channels, height and width of the input.
        input_dim: (NonZeroUsize, NonZeroUsize, NonZeroUsize),
        /// The filters, in channels, and size of the square kernel.
        kernel_dim: (NonZeroUsize, NonZeroUsize, NonZeroUsize),
        stride: NonZeroUsize,
        padding: usize,
        init: ParamGenConfig,
        #[serde(default)]
        act_fn: Option<ActFnConfig>,
    },
}

/// The `Model` configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ModelConfig {
    pub layers: Vec<LayerConfig>,
}
