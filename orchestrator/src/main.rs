use std::{
    env, io,
    num::NonZeroUsize,
    process::{Command, ExitStatus},
};

use comms::Float01;
use orchestrator::{CancelHandle, configs::*, train};

const MODEL_OUTPUT_PATH: &str = "model.safetensors";
const SERVER_BASE_PORT: usize = 40_000;
const WORKER_BASE_PORT: usize = 50_000;

// The file path for the compose up script file.
const COMPOSE_FILE_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../docker/compose_up.py");

/// Sets up the docker containers for simulated execution of the system.
///
/// # Args
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
/// # Args
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
    const WORKERS: usize = 3;
    const SERVERS: usize = 0;
    const RELEASE: bool = false;

    setup_docker(WORKERS, SERVERS, RELEASE)?;
    let (worker_addrs, server_addrs) = build_addresses(WORKERS, SERVERS);

    let model_config = ModelConfig {
        layers: vec![
            LayerConfig::Conv {
                input_dim: (
                    NonZeroUsize::new(2).unwrap(),
                    NonZeroUsize::new(3).unwrap(),
                    NonZeroUsize::new(3).unwrap(),
                ),
                kernel_dim: (
                    NonZeroUsize::new(5).unwrap(),
                    NonZeroUsize::new(2).unwrap(),
                    NonZeroUsize::new(2).unwrap(),
                ),
                stride: NonZeroUsize::new(1).unwrap(),
                padding: 0,
                init: ParamGenConfig::Kaiming,
                act_fn: None,
            },
            LayerConfig::Dense {
                output_size: NonZeroUsize::new(4).unwrap(),
                init: ParamGenConfig::Kaiming,
                act_fn: Some(ActFnConfig::Softmax),
            },
        ],
    };

    let algorithm_config = if !server_addrs.is_empty() {
        AlgorithmConfig::ParameterServer {
            server_addrs,
            synchronizer: SynchronizerConfig::NonBlocking,
            store: StoreConfig::Wild,
        }
    } else {
        AlgorithmConfig::AllReduce
    };

    let training_config = TrainingConfig {
        worker_addrs,
        algorithm: algorithm_config,
        serializer: SerializerConfig::SparseCapable {
            r: Float01::new(0.9).unwrap(),
        },
        dataset: DatasetConfig {
            src: DatasetSrc::Inline {
                samples: vec![
                    0.0, 1.0, 0.0, //
                    1.0, 1.0, 1.0, //
                    0.0, 1.0, 0.0, //
                    //
                    1.0, 0.0, 1.0, //
                    0.0, 0.0, 0.0, //
                    1.0, 0.0, 1.0, // plus sign
                    //
                    0.0, 0.0, 0.0, //
                    0.0, 1.0, 0.0, //
                    0.0, 0.0, 0.0, //
                    //
                    1.0, 1.0, 1.0, //
                    1.0, 0.0, 1.0, //
                    1.0, 1.0, 1.0, // dot
                    //
                    1.0, 0.0, 1.0, //
                    0.0, 1.0, 0.0, //
                    1.0, 0.0, 1.0, //
                    //
                    0.0, 1.0, 0.0, //
                    1.0, 0.0, 1.0, //
                    0.0, 1.0, 0.0, // cross
                    //
                    1.0, 1.0, 1.0, //
                    1.0, 0.0, 1.0, //
                    1.0, 1.0, 1.0, //
                    //
                    0.0, 0.0, 0.0, //
                    0.0, 1.0, 0.0, //
                    0.0, 0.0, 0.0, // box
                ],
                labels: vec![
                    1.0, 0.0, 0.0, 0.0, // plus sign
                    0.0, 1.0, 0.0, 0.0, // dot
                    0.0, 0.0, 1.0, 0.0, // cross
                    0.0, 0.0, 0.0, 1.0, // box
                ],
            },
            x_size: NonZeroUsize::new(18).unwrap(),
            y_size: NonZeroUsize::new(4).unwrap(),
        },
        optimizer: OptimizerConfig::GradientDescentWithMomentum {
            lr: 1.0,
            mu: Float01::new(0.9).unwrap(),
        },
        loss_fn: LossFnConfig::Mse,
        batch_size: NonZeroUsize::new(4).unwrap(),
        max_epochs: NonZeroUsize::new(1000).unwrap(),
        offline_epochs: 0,
        seed: Some(42),
        early_stopping: None,
    };

    let session = train(model_config, training_config).unwrap();
    let (_cancel, cancel_rx) = CancelHandle::pair();
    let mut rx = session.event_listener(cancel_rx);

    loop {
        match rx.blocking_recv() {
            Some(orchestrator::TrainingEvent::Loss { losses, worker_id }) => {
                println!("losses {worker_id}: {losses:?}")
            }
            Some(orchestrator::TrainingEvent::Complete {
                model: trained,
                reason,
            }) => {
                println!("params: {:?}", trained.params());
                println!("stop reason: {reason:?}");

                trained
                    .save_safetensors(MODEL_OUTPUT_PATH)
                    .expect("failed to save model");

                println!("saved model to {MODEL_OUTPUT_PATH}");
                break;
            }
            None => break,
            res => println!("result: {res:?}"),
        }
    }

    Ok(())
}
