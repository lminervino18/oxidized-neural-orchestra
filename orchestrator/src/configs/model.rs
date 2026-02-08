#[derive(Debug, Clone, Copy)]
pub enum ActFnConfig {
    Sigmoid { amp: f32 },
}

#[derive(Debug, Clone, Copy)]
pub enum ParamGenConfig {
    Const {
        value: f32,
        limit: usize,
    },
    Uniform {
        low: f32,
        high: f32,
        limit: usize,
    },
    UniformInclusive {
        low: f32,
        high: f32,
        limit: usize,
    },
    XavierUniform {
        fan_in: usize,
        fan_out: usize,
        limit: usize,
    },
    LecunUniform {
        fan_in: usize,
        limit: usize,
    },
    Normal {
        mean: f32,
        std_dev: f32,
        limit: usize,
    },
    Kaiming {
        fan_in: usize,
        limit: usize,
    },
    Xavier {
        fan_in: usize,
        fan_out: usize,
        limit: usize,
    },
    Lecun {
        fan_in: usize,
        limit: usize,
    },
}

#[derive(Debug, Clone, Copy)]
pub enum LayerConfig {
    Dense {
        dim: (usize, usize),
        init: ParamGenConfig,
        act_fn: Option<ActFnConfig>,
    },
}

#[derive(Debug)]
pub enum ModelConfig {
    Sequential { layers: Vec<LayerConfig> },
}
