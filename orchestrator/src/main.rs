use std::num::NonZeroUsize;

use orchestrator::{configs::*, train};

fn main() {
    env_logger::init();

    let model_config = ModelConfig::Sequential {
        layers: vec![LayerConfig::Dense {
            dim: (1, 1),
            init: ParamGenConfig::Const { value: 0.0 },
            act_fn: None,
        }],
    };

    let training_config = TrainingConfig {
        worker_addrs: vec!["worker-0:50000"],
        algorithm: AlgorithmConfig::ParameterServer {
            server_addrs: vec!["server-0:40000"],
            synchronizer: SynchronizerConfig::Barrier { barrier_size: 1 },
            store: StoreConfig::Blocking {
                shard_size: NonZeroUsize::new(1).expect("1 is non-zero"),
            },
        },
        dataset: DatasetConfig::Inline {
            data: vec![
                1.0, 2.0,
                2.0, 4.0,
                3.0, 6.0,
                4.0, 8.0,
            ],
            x_size: 1,
            y_size: 1,
        },
        optimizer: OptimizerConfig::GradientDescent { lr: 0.01 },
        loss_fn: LossFnConfig::Mse,
        batch_size: NonZeroUsize::new(4).expect("non-zero"),
        max_epochs: NonZeroUsize::new(700).expect("non-zero"),
        offline_epochs: 0,
        seed: Some(42),
    };

    log::info!("starting distributed training session");

    match train(model_config, training_config) {
        Err(e) => log::error!("failed to start session: {e}"),
        Ok(session) => {
            log::info!("session started, waiting for completion");
            match session.wait() {
                Ok(params) => {
                    log::info!("training complete");
                    println!("trained params: {params:?}");
                }
                Err(e) => log::error!("training failed: {e}"),
            }
        }
    }
}