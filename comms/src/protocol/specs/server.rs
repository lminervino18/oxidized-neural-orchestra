use std::num::NonZeroUsize;

use serde::{Deserialize, Serialize};

use super::machine_learning::{OptimizerSpec, ParamGenSpec};

/// The specification for the `Synchronizer` trait.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SynchronizerSpec {
    Barrier { barrier_size: NonZeroUsize },
    NonBlocking,
}

/// The specification for the `Store` trait.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoreSpec {
    Blocking,
    Wild,
}

/// The specification for the `Server` trait.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerSpec {
    pub nworkers: usize,
    pub param_gen: ParamGenSpec,
    pub optimizer: OptimizerSpec,
    pub synchronizer: SynchronizerSpec,
    pub store: StoreSpec,
    pub seed: Option<u64>,
}
