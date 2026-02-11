use std::num::NonZeroUsize;

use crate::configs::{
    AlgorithmConfig, DatasetConfig, LossFnConfig, OptimizerConfig, StoreConfig, SynchronizerConfig,
    TrainingConfig,
};

pub fn parameter_server<Sy: Into<Option<SynchronizerConfig>>, PS: Into<Option<StoreConfig>>>(
    worker_addrs: Vec<String>,
    server_addrs: Vec<String>,
    synchronizer: Sy,
    store: PS,
    dataset: DatasetConfig,
    optimizer: OptimizerConfig,
    loss_fn: LossFnConfig,
    epochs: NonZeroUsize,
    batch_size: NonZeroUsize,
    seed: Option<u64>,
) -> TrainingConfig {
    let synchronizer = synchronizer
        .into()
        .unwrap_or_else(|| SynchronizerConfig::Barrier {
            barrier_size: worker_addrs.len(),
        });

    let store = store.into().unwrap_or_else(|| StoreConfig::Blocking {
        shard_size: NonZeroUsize::new(1).unwrap(), // TODO: Ver como definir bien esto, sino que sea obligatorio el store
    });

    TrainingConfig {
        worker_addrs,
        algorithm: AlgorithmConfig::ParameterServer {
            server_addrs,
            synchronizer,
            store,
        },
        dataset,
        optimizer,
        loss_fn,
        epochs,
        batch_size,
        seed,
    }
}
