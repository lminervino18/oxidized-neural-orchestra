use std::{net::SocketAddr, num::NonZeroUsize};

use crate::configs::{
    AlgorithmConfig, DatasetConfig, LossFnConfig, OptimizerConfig, StoreConfig, SynchronizerConfig,
    TrainingConfig,
};

pub fn parameter_server<Sy: Into<Option<SynchronizerConfig>>, PS: Into<Option<StoreConfig>>>(
    worker_ips: Vec<SocketAddr>,
    server_ips: Vec<SocketAddr>,
    synchronizer: Sy,
    store: PS,
    dataset: DatasetConfig,
    optimizer: OptimizerConfig,
    loss_fn: LossFnConfig,
    offline_epochs: usize,
    batch_size: NonZeroUsize,
    seed: Option<u64>,
) -> TrainingConfig {
    let synchronizer = synchronizer
        .into()
        .unwrap_or_else(|| SynchronizerConfig::Barrier {
            barrier_size: worker_ips.len(),
        });

    let store = store.into().unwrap_or_else(|| StoreConfig::Blocking {
        shard_size: NonZeroUsize::new(1).unwrap(), // TODO: Ver como definir bien esto, sino que sea obligatorio el store
    });

    TrainingConfig {
        worker_ips,
        algorithm: AlgorithmConfig::ParameterServer {
            server_ips,
            synchronizer,
            store,
        },
        dataset,
        optimizer,
        loss_fn,
        offline_epochs,
        batch_size,
        seed,
    }
}
