mod adapter;
mod model;
mod partition;
mod training;
mod validator;

pub use adapter::Adapter;
pub use model::{ActFnConfig, LayerConfig, ModelConfig, ParamGenConfig};
pub use partition::Partition;
pub use training::{
    AlgorithmConfig, DataSrc, DatasetConfig, EarlyStoppingConfig, LossFnConfig, OptimizerConfig,
    SerializerConfig, StoreConfig, SynchronizerConfig, TrainingConfig,
};
pub use validator::Validator;

use comms::specs::{server::ServerSpec, worker::WorkerSpec};

// TODO: docstring
#[derive(Debug, Clone)]
pub enum WorkerPostAction {
    Switch {
        server_addrs: Vec<String>,
        server_sizes: Vec<usize>,
        server_ordering: Vec<usize>,
    },
    Upgrade {
        spec: ServerSpec,
    },
}

/// The result of adapting a worker configuration.
#[derive(Debug, Clone)]
pub struct WorkerAdapt<'a> {
    pub addr: String,
    pub spec: WorkerSpec,
    pub partition: Partition<'a>,
    pub post_action: Option<WorkerPostAction>,
}

/// The result of adapting a server configuration.
#[derive(Debug, Clone)]
pub struct ServerAdapt {
    pub addr: String,
    pub spec: ServerSpec,
}
