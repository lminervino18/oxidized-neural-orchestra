use orchestrator::{configs::*, train};
use std::{
    env, io,
    num::NonZeroUsize,
    process::{Command, ExitStatus},
};

const MODEL_OUTPUT_PATH: &str = "model.safetensors";
const SERVER_BASE_PORT: usize = 40_000;
const WORKER_BASE_PORT: usize = 50_000;

// The file path for the compose up script file.
const COMPOSE_FILE_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../docker/compose_up.py");

/// Set ups the docker containers for simulated execution of the system.
///
/// # Arguments
/// * `workers` - The amount of workers to use.
/// * `servers` - The amount of servers to use.
/// * `release` - The compilation mode for the rust compiler.
///
/// # Returns
/// The exit status of the compose script.
fn setup_docker(workers: usize, servers: usize, release: bool) -> io::Result<ExitStatus> {
    let mut cmd = Command::new("python3");

    cmd.arg(COMPOSE_FILE_PATH)
        .arg("--workers")
        .arg(workers.to_string())
        .arg("--servers")
        .arg(servers.to_string());

    if release {
        cmd.arg("--release");
    }

    cmd.spawn()?.wait()
}

/// Builds the addresses for both workers and servers.
///
/// # Arguments
/// * `workers` - The amount of workers to use.
/// * `servers` - The amount of servers to use.
///
/// # Returns
/// A tuple of lists of addresses.
fn build_addresses(workers: usize, servers: usize) -> (Vec<String>, Vec<String>) {
    let worker_addrs = (0..workers)
        .map(|i| format!("worker-{i}:{}", WORKER_BASE_PORT + i))
        .collect();

    let server_addrs = (0..servers)
        .map(|i| format!("server-{i}:{}", SERVER_BASE_PORT + i))
        .collect();

    (worker_addrs, server_addrs)
}

fn main() -> io::Result<()> {
    let workers = 3;
    let servers = 2;
    let release = false;

    setup_docker(workers, servers, release)?;
    let (worker_addrs, server_addrs) = build_addresses(workers, servers);

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

    let training_config = TrainingConfig {
        worker_addrs,
        algorithm: AlgorithmConfig::ParameterServer {
            server_addrs,
            synchronizer: SynchronizerConfig::Barrier,
            store: StoreConfig::Blocking,
        },
        dataset: DatasetConfig {
            src: DatasetSrc::Inline {
                data: vec![
                    0.0, 0.0, 0.0, //
                    0.0, 1.0, 1.0, //
                    1.0, 0.0, 1.0, //
                    1.0, 1.0, 0.0,
                ],
            },
            x_size: NonZeroUsize::new(2).unwrap(),
            y_size: NonZeroUsize::new(1).unwrap(),
        },
        optimizer: OptimizerConfig::GradientDescent { lr: 1.0 },
        loss_fn: LossFnConfig::Mse,
        batch_size: NonZeroUsize::new(4).unwrap(),
        max_epochs: NonZeroUsize::new(500).unwrap(),
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

                trained
                    .save_safetensors(MODEL_OUTPUT_PATH)
                    .expect("failed to save model");

                println!("saved model to {MODEL_OUTPUT_PATH}");
                break;
            }
            res => println!("result: {res:?}"),
        }
    }

    Ok(())
}
