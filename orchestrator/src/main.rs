use std::{num::NonZeroUsize, path::PathBuf};

use orchestrator::{configs::*, train};

fn main() {
    env_logger::init();

    let model_config = ModelConfig {
        layers: vec![
            LayerConfig::Dense {
                output_size: NonZeroUsize::new(2).unwrap(),
                init: ParamGenConfig::Kaiming,
                act_fn: Some(ActFnConfig::Sigmoid { amp: 1.0 }),
            },
            LayerConfig::Dense {
                output_size: NonZeroUsize::new(1).unwrap(),
                init: ParamGenConfig::Kaiming,
                act_fn: Some(ActFnConfig::Sigmoid { amp: 1.0 }),
            },
        ],
    };

    let path = PathBuf::from("dataset");
    let training_config = TrainingConfig {
        worker_addrs: vec!["worker-0:50000"],
        algorithm: AlgorithmConfig::ParameterServer {
            server_addrs: vec!["server-0:40000", "server-1:40001"],
            synchronizer: SynchronizerConfig::Barrier,
            store: StoreConfig::Blocking,
        },
        dataset: DatasetConfig {
            src: DatasetSrc::Local { path },
            x_size: NonZeroUsize::new(2).unwrap(),
            y_size: NonZeroUsize::new(1).unwrap(),
        },
        optimizer: OptimizerConfig::GradientDescent { lr: 1.0 },
        loss_fn: LossFnConfig::Mse,
        batch_size: NonZeroUsize::new(4).unwrap(),
        max_epochs: NonZeroUsize::new(1000).unwrap(),
        offline_epochs: 0,
        seed: Some(42),
    };

    let session = train(model_config, training_config).unwrap();
    let mut rx = session.event_listener();

    loop {
        match rx.blocking_recv() {
            Some(orchestrator::TrainingEvent::Loss { losses, .. }) => {
                println!("losses: {losses:?}")
            }
            Some(orchestrator::TrainingEvent::Complete(params)) => {
                println!("params: {params:?}");
                break;
            }
            res => println!("result: {res:?}"),
        }
    }
}
