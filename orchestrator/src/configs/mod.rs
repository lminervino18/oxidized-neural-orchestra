mod adapter;
mod model;
mod training;

pub use adapter::to_specs_adapter;
pub(super) use model::{ActFnConfig, LayerConfig, ModelConfig, ParamGenConfig};
pub(super) use training::{
    AlgorithmConfig, DatasetConfig, LossFnConfig, OptimizerConfig, StoreConfig, SynchronizerConfig,
    TrainingConfig,
};
