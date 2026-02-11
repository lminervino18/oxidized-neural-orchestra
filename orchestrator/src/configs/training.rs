use std::{
    net::{SocketAddr, ToSocketAddrs},
    num::NonZeroUsize,
    path::PathBuf,
};

#[derive(Debug, Clone, Copy)]
pub enum LossFnConfig {
    Mse,
}

#[derive(Debug, Clone, Copy)]
pub enum OptimizerConfig {
    Adam { lr: f32, b1: f32, b2: f32, eps: f32 },
    GradientDescent { lr: f32 },
    GradientDescentWithMomentum { lr: f32, mu: f32 },
}

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

#[derive(Debug, Clone, Copy)]
pub enum SynchronizerConfig {
    Barrier { barrier_size: usize },
    NonBlocking,
}

#[derive(Debug, Clone, Copy)]
pub enum StoreConfig {
    Blocking { shard_size: NonZeroUsize },
    Wild { shard_size: NonZeroUsize },
}

#[derive(Debug)]
pub enum AlgorithmConfig {
    ParameterServer {
        server_addrs: Vec<String>,
        synchronizer: SynchronizerConfig,
        store: StoreConfig,
    },
}

#[derive(Debug)]
pub struct TrainingConfig {
    pub worker_addrs: Vec<String>,
    pub algorithm: AlgorithmConfig,
    pub dataset: DatasetConfig,
    pub optimizer: OptimizerConfig,
    pub loss_fn: LossFnConfig,
    pub epochs: NonZeroUsize,
    pub batch_size: NonZeroUsize,
    pub seed: Option<u64>,
}
