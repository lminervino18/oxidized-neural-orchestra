use std::num::NonZeroUsize;

use comms::floats::Float01;
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
    Tanh { amp: f32 },
    ReLU { slope: Float01 },
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

impl LayerConfig {
    /// The amount of values this layer produces.
    ///
    /// # Returns
    /// The layer's output size.
    pub fn output_size(&self) -> NonZeroUsize {
        match *self {
            LayerConfig::Dense { output_size, .. } => output_size,
            LayerConfig::Conv {
                input_dim,
                kernel_dim,
                stride,
                padding,
                ..
            } => {
                let (filters, _, kernel_size) =
                    (kernel_dim.0.get(), kernel_dim.1.get(), kernel_dim.2.get());
                let output_height =
                    (input_dim.1.get() + 2 * padding).saturating_sub(kernel_size) / stride.get() + 1;
                let output_width =
                    (input_dim.2.get() + 2 * padding).saturating_sub(kernel_size) / stride.get() + 1;

                NonZeroUsize::new(output_height * output_width * filters).unwrap()
            }
        }
    }

    /// The flattened input size this layer expects, when it is fixed by the layer itself.
    ///
    /// # Returns
    /// `Some(size)` for layers with a fixed input shape (`Conv`), `None` otherwise.
    pub fn expected_input_size(&self) -> Option<NonZeroUsize> {
        match *self {
            LayerConfig::Conv { input_dim, .. } => {
                NonZeroUsize::new(input_dim.0.get() * input_dim.1.get() * input_dim.2.get())
            }
            LayerConfig::Dense { .. } => None,
        }
    }
}

/// The `Model` configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ModelConfig {
    pub layers: Vec<LayerConfig>,
}
