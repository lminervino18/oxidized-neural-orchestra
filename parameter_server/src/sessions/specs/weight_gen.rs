#[derive(Clone, Copy)]
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

pub enum WeightGenSpec {
    Const {
        value: f32,
        limit: usize,
    },
    Rand {
        distribution: DistributionSpec,
        limit: usize,
    },
    Chained {
        specs: Vec<WeightGenSpec>,
    },
}
