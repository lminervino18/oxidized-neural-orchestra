use std::num::NonZeroUsize;

use orchestrator::{configs::*, train};

fn main() {
    let model_config = ModelConfig::Sequential {
        layers: vec![LayerConfig::Dense {
            dim: (1, 1),
            init: ParamGenConfig::Const {
                value: 0.0,
                limit: 1,
            },
            act_fn: None,
        }],
    };

    let training_config = TrainingConfig {
        worker_addrs: vec!["127.0.0.1:8764".into()],
        algorithm: AlgorithmConfig::ParameterServer {
            server_addrs: vec!["127.0.0.1:8765".into()],
            synchronizer: SynchronizerConfig::Barrier { barrier_size: 1 },
            store: StoreConfig::Blocking {
                shard_size: NonZeroUsize::new(1).unwrap(),
            },
        },
        dataset: DatasetConfig::Inline {
            data: vec![0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0],
            x_size: 1,
            y_size: 1,
        },
        optimizer: OptimizerConfig::GradientDescent { lr: 0.1 },
        loss_fn: LossFnConfig::Mse,
        epochs: NonZeroUsize::new(100).unwrap(),
        batch_size: NonZeroUsize::new(4).unwrap(),
        seed: None,
    };

    let session = train(model_config, training_config).unwrap();
    let params = session.wait().unwrap();
    println!("trained params: {params:?}");
}
