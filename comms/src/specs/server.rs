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

/// The specification for the `WeightGen` trait.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
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

/// The specification for the `Trainer` trait.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainerSpec {
    BarrierSync { barrier_size: usize },
    NonBlocking,
}

/// The specification for the `Server` trait.
#[derive(Debug, Serialize, Deserialize)]
pub struct ServerSpec {
    pub workers: usize,
    pub shard_size: NonZeroUsize,
    pub weight_gen: WeightGenSpec,
    pub optimizer: OptimizerSpec,
    pub trainer: TrainerSpec,
    pub seed: Option<u64>,
}
