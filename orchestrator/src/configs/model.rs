/// The `ParamGen` configuration.
#[derive(Debug, Clone, Copy)]
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
#[derive(Debug, Clone, Copy)]
pub enum ActFnConfig {
    Sigmoid { amp: f32 },
}

/// The `Layer` configuration.
#[derive(Debug, Clone, Copy)]
pub enum LayerConfig {
    Dense {
        dim: (usize, usize),
        init: ParamGenConfig,
        act_fn: Option<ActFnConfig>,
    },
}

impl LayerConfig {
    /// Obtains the fan_in, size and fan_out of the layer.
    ///
    /// # Returns
    /// A tuple (fan_in, size, fan_out).
    pub fn sizes(&self) -> (usize, usize, usize) {
        match *self {
            LayerConfig::Dense { dim: (n, m), .. } => (n, n * m + m, m),
        }
    }
}

/// The `Model` configuration.
#[derive(Debug)]
pub enum ModelConfig {
    Sequential { layers: Vec<LayerConfig> },
}
