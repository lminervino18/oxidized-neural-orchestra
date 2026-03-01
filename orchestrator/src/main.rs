use std::num::NonZeroUsize;

use orchestrator::{configs::*, train};

fn main() {
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
                0.0, 1.0,  1.0, 2.0,  2.0, 3.0,  3.0, 4.0,
                4.0, 5.0,  5.0, 6.0,  6.0, 7.0,  7.0, 8.0,
                8.0, 9.0,  9.0, 10.0, 10.0, 11.0, 11.0, 12.0,
                12.0, 13.0, 13.0, 14.0, 14.0, 15.0, 15.0, 16.0,
            ],
            x_size: 1,
            y_size: 1,
        },
        optimizer: OptimizerConfig::GradientDescent { lr: 0.1 },
        loss_fn: LossFnConfig::Mse,
        batch_size: NonZeroUsize::new(4).expect("4 is non-zero"),
        max_epochs: NonZeroUsize::new(100).expect("100 is non-zero"),
        offline_epochs: 0,
        seed: None,
    };

    match train(model_config, training_config) {
        Err(e) => eprintln!("failed to start session: {e}"),
        Ok(session) => match session.wait() {
            Ok(params) => println!("trained params: {params:?}"),
            Err(e) => eprintln!("training failed: {e}"),
        },
    }
}