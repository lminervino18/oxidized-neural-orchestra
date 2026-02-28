use std::{net::ToSocketAddrs, num::NonZeroUsize, path::PathBuf};

/// The `LossFn` configuration.
#[derive(Debug, Clone, Copy)]
pub enum LossFnConfig {
    Mse,
}

/// The `Optimizer` configuration.
#[derive(Debug, Clone, Copy)]
pub enum OptimizerConfig {
    Adam { lr: f32, b1: f32, b2: f32, eps: f32 },
    GradientDescent { lr: f32 },
    GradientDescentWithMomentum { lr: f32, mu: f32 },
}

/// The `Dataset` configuration.
#[derive(Debug)]
pub enum DatasetConfig {
    Local {
        path: PathBuf,
    },
    Inline {
        data: Vec<f32>,
        x_size: usize,
        y_size: usize,
    },
}

/// The `Synchronizer` configuration.
#[derive(Debug, Clone, Copy)]
pub enum SynchronizerConfig {
    Barrier { barrier_size: usize },
    NonBlocking,
}

/// The `Store` configuration.
#[derive(Debug, Clone, Copy)]
pub enum StoreConfig {
    Blocking { shard_size: NonZeroUsize },
    Wild { shard_size: NonZeroUsize },
}

/// The `Algorithm` configuration.
#[derive(Debug)]
pub enum AlgorithmConfig<A: ToSocketAddrs> {
    ParameterServer {
        server_addrs: Vec<A>,
        synchronizer: SynchronizerConfig,
        store: StoreConfig,
    },
}

/// The `Training` configuration.
#[derive(Debug)]
pub struct TrainingConfig<A: ToSocketAddrs> {
    pub worker_addrs: Vec<A>,
    pub algorithm: AlgorithmConfig<A>,
    pub dataset: DatasetConfig,
    pub optimizer: OptimizerConfig,
    pub loss_fn: LossFnConfig,
    pub batch_size: NonZeroUsize,
    pub max_epochs: NonZeroUsize,
    pub offline_epochs: usize,
    pub seed: Option<u64>,
}
