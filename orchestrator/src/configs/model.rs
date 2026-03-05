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
}

/// The `Layer` configuration.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerConfig {
    Dense {
        output_size: NonZeroUsize,
        init: ParamGenConfig,
        act_fn: Option<ActFnConfig>,
    },
}

/// The `Model` configuration.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelConfig {
    Sequential { layers: Vec<LayerConfig> },
}
