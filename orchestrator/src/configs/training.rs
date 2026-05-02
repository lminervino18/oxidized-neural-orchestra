use std::{fmt, num::NonZeroUsize, ops::Deref, path::PathBuf};

use comms::Float01;
use serde::{Deserialize, Deserializer, Serialize, de};

#[derive(Serialize, Debug, Clone, Copy)]
#[serde(transparent)]
pub struct FloatPositive {
    value: f32,
}

impl FloatPositive {
    /// Creates a new `FloatPositive`.
    ///
    /// # Args
    /// * `value` - The inner value.
    ///
    /// # Returns
    /// An option with `Some` value if the given `value` is a positive number, else `None`.
    pub fn new(value: f32) -> Option<Self> {
        value.is_sign_positive().then_some(FloatPositive { value })
    }
}

impl<'de> Deserialize<'de> for FloatPositive {
    /// Deserializes a `FloatPositive` from a float value.
    ///
    /// # Args
    /// * `deserializer` - The deserializer to read from.
    ///
    /// # Returns
    /// A validated `Float01` instance.
    ///
    /// # Errors
    /// Returns a deserialization error if the value is outside `[0.0, 1.0]`.
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = f32::deserialize(deserializer)?;
        FloatPositive::new(value)
            .ok_or_else(|| de::Error::custom("Float01 value must be between 0.0 and 1.0"))
    }
}

impl Deref for FloatPositive {
    type Target = f32;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

/// Criteria for stopping training early when loss improvement falls below a threshold.
///
/// Guarantees that `tolerance` is strictly positive, which is enforced at construction time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingConfig {
    pub tolerance: FloatPositive,
}

impl fmt::Display for EarlyStoppingConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.2e}", *self.tolerance)
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
    GradientDescentWithMomentum { lr: f32, mu: Float01 },
    Adam { lr: f32, b1: f32, b2: f32, eps: f32 },
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
