use std::{fmt, num::NonZeroUsize, path::PathBuf};

use comms::Float01;
use serde::{Deserialize, Serialize};

/// Criteria for stopping training early when loss improvement falls below a threshold.
///
/// Guarantees that `tolerance` is strictly positive, which is enforced at construction time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingConfig {
    tolerance: f32,
}

impl EarlyStoppingConfig {
    pub fn new(tolerance: f32) -> Result<Self, String> {
        if tolerance <= 0.0 {
            return Err(format!(
                "early_stopping tolerance must be > 0, got {tolerance}"
            ));
        }
        Ok(Self { tolerance })
    }

    pub fn is_converged(&self, prev: f32, curr: f32) -> bool {
        (prev - curr).abs() < self.tolerance
    }
}

impl fmt::Display for EarlyStoppingConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.2e}", self.tolerance)
    }
}

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
    AllReduce,
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
    #[serde(default)]
    pub early_stopping: Option<EarlyStoppingConfig>,
}
