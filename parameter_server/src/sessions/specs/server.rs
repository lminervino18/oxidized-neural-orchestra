use std::num::NonZeroUsize;

use super::{OptimizerSpec, TrainerSpec, WeightGenSpec};

pub struct ParameterServerSpec {
    pub params: NonZeroUsize,
    pub shard_amount: Option<NonZeroUsize>,
    pub epochs: usize,
    pub weight_gen: WeightGenSpec,
    pub optimizer: OptimizerSpec,
    pub trainer: TrainerSpec,
    pub seed: Option<u64>,
}
