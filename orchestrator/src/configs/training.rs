use std::{num::NonZeroUsize, path::PathBuf};

use comms::Float01;
use serde::{Deserialize, Serialize};

/// The `LossFn` configuration.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LossFnConfig {
    Mse,
    CrossEntropy,
}

/// The `Optimizer` configuration.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OptimizerConfig {
    GradientDescent { lr: f32 },
}

/// The dataset's data source.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DatasetSrc {
    Local {
        samples_path: PathBuf,
        labels_path: PathBuf,
    },
    Inline {
        samples: Vec<f32>,
        labels: Vec<f32>,
    },
}

/// The `Dataset` configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct DatasetConfig {
    pub src: DatasetSrc,
    pub x_size: NonZeroUsize,
    pub y_size: NonZeroUsize,
}

/// The `Synchronizer` configuration.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SynchronizerConfig {
    Barrier,
    NonBlocking,
}

/// The `Store` configuration.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoreConfig {
    Blocking,
    Wild,
}

/// The `Algorithm` configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlgorithmConfig {
    ParameterServer {
        server_addrs: Vec<String>,
        synchronizer: SynchronizerConfig,
        store: StoreConfig,
    },
    RingAllReduce,
}

/// The `Serializer` configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SerializerConfig {
    Base,
    SparseCapable { r: Float01 },
}

impl Default for SerializerConfig {
    fn default() -> Self {
        Self::Base
    }
}

/// The `Training` configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub worker_addrs: Vec<String>,
    pub algorithm: AlgorithmConfig,
    #[serde(default)]
    pub serializer: SerializerConfig,
    pub dataset: DatasetConfig,
    pub optimizer: OptimizerConfig,
    pub loss_fn: LossFnConfig,
    pub batch_size: NonZeroUsize,
    pub max_epochs: NonZeroUsize,
    pub offline_epochs: usize,
    #[serde(default)]
    pub seed: Option<u64>,
}
