mod adapter;
mod model;
mod partition;
mod training;
mod validator;

use std::num::NonZeroUsize;

use comms::specs::{server::ServerSpec, worker::WorkerSpec};

pub use adapter::Adapter;
pub use model::{ActFnConfig, LayerConfig, ModelConfig, ParamGenConfig};
pub use partition::Partition;
pub use training::{
    AlgorithmConfig, DataSrc, DatasetConfig, EarlyStoppingConfig, LossFnConfig, OptimizerConfig,
    SerializerConfig, StoreConfig, SynchronizerConfig, TrainingConfig,
};
pub use validator::Validator;

use crate::sessions::{ConvergenceTracker, LossRecorder, SwitchTracker};

/// An action taken by the orchestrator based on strategy switch for each worker.
#[derive(Debug, Clone)]
pub enum WorkerPostAction {
    Switch {
        server_addrs: Vec<String>,
        server_sizes: Vec<usize>,
        server_ordering: Vec<usize>,
    },
    Upgrade {
        spec: ServerSpec,
        ranges: Vec<(usize, usize)>,
    },
}

/// The adapted values for worker initialization.
#[derive(Debug, Clone)]
pub struct WorkerAdapt<'a> {
    pub addr: String,
    pub spec: WorkerSpec,
    pub partition: Partition<'a>,
}

/// The adapted values for server initialization.
#[derive(Debug, Clone)]
pub struct ServerAdapt {
    pub addr: String,
    pub spec: ServerSpec,
}

/// The metadata to hold for strategy switching.
#[derive(Debug)]
pub struct StrategySwitchTracking {
    pub tracker: SwitchTracker,
    pub post_actions: Vec<WorkerPostAction>,
}

/// The adaptated values for session initialization.
#[derive(Debug)]
pub struct OrchAdapt {
    pub input_size: NonZeroUsize,
    pub loss_recorder: LossRecorder,
    pub convergence_tracker: Option<ConvergenceTracker>,
    pub switch_tracking: Option<StrategySwitchTracking>,
    pub model_config: ModelConfig,
    pub algorithm_config: AlgorithmConfig,
}
