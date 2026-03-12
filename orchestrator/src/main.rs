use std::{env, num::NonZeroUsize, path::PathBuf};

use orchestrator::{configs::*, train};

/// Reads a required environment variable, panicking with a clear message if absent.
///
/// # Arguments
/// * `key` - The name of the environment variable.
///
/// # Returns
/// The value of the environment variable as a `String`.
///
/// # Panics
/// Panics if the variable is not set.
fn require_env(key: &str) -> String {
    env::var(key).unwrap_or_else(|_| panic!("required env var {key} is not set"))
}

/// Parses a required environment variable as a `usize`, panicking if absent or unparseable.
///
/// # Arguments
/// * `key` - The name of the environment variable.
///
/// # Returns
/// The parsed `usize` value.
///
/// # Panics
/// Panics if the variable is not set or cannot be parsed as `usize`.
fn require_env_usize(key: &str) -> usize {
    require_env(key)
        .parse::<usize>()
        .unwrap_or_else(|_| panic!("env var {key} must be a valid usize"))
}

fn main() {
    env_logger::init();

    let nworkers = require_env_usize("WORKERS");
    let nservers = require_env_usize("SERVERS");

    let worker_addrs: Vec<String> = (0..nworkers)
        .map(|i| format!("worker-{i}:{}", 50_000 + i))
        .collect();

    let server_addrs: Vec<String> = (0..nservers)
        .map(|i| format!("server-{i}:{}", 40_000 + i))
        .collect();

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

    let path = PathBuf::from("/dataset");
    let training_config = TrainingConfig {
        worker_addrs,
        algorithm: AlgorithmConfig::ParameterServer {
            server_addrs,
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
        max_epochs: NonZeroUsize::new(100).unwrap(),
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
            Some(orchestrator::TrainingEvent::Complete(trained)) => {
                println!("params: {:?}", trained.params());

                let out_path = "model.safetensors";
                trained
                    .save_safetensors(out_path)
                    .expect("failed to save model");

                let size = std::fs::metadata(out_path)
                    .expect("model.safetensors not found after save")
                    .len();
                assert!(size > 0, "model.safetensors is empty after save");
                println!("model saved to {out_path} ({size} bytes)");

                break;
            }
            res => println!("result: {res:?}"),
        }
    }
}
