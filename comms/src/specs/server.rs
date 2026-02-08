use std::num::NonZeroUsize;

use serde::{Deserialize, Serialize};

use super::machine_learning::OptimizerSpec;

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
#[derive(Debug, Serialize, Deserialize)]
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

/// The specification for the `Synchronizer` trait.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SynchronizerSpec {
    Barrier { barrier_size: usize },
    NonBlocking,
}

/// The specification for the `Store` trait.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoreSpec {
    Blocking { shard_size: NonZeroUsize },
    Wild { shard_size: NonZeroUsize },
}

/// The specification for the `Server` trait.
#[derive(Debug, Serialize, Deserialize)]
pub struct ServerSpec {
    pub nworkers: usize,
    pub param_gen: ParamGenSpec,
    pub optimizer: OptimizerSpec,
    pub synchronizer: SynchronizerSpec,
    pub store: StoreSpec,
    pub seed: Option<u64>,
}
