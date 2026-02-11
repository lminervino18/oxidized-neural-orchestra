mod adapter;
mod model;
mod training;

pub use adapter::to_specs_adapter;
pub use model::{ActFnConfig, LayerConfig, ModelConfig, ParamGenConfig};
pub use training::{
    AlgorithmConfig, DatasetConfig, LossFnConfig, OptimizerConfig, StoreConfig, SynchronizerConfig,
    TrainingConfig,
};
