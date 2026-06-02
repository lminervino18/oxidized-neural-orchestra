use std::{
    env, io,
    num::NonZeroUsize,
    process::{Command, ExitStatus},
    thread,
    time::Duration,
};

use comms::floats::{Float01, FloatPositive};
use log::info;
use orchestrator::{CancelHandle, configs::*, train};

const MODEL_OUTPUT_PATH: &str = "model.safetensors";
const NODE_BASE_PORT: usize = 40_000;

// The file path for the compose up script file.
const COMPOSE_FILE_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../docker/compose_up.py");

/// Sets up the docker containers for simulated execution of the system.
///
/// # Args
/// * `nodes` - The amount of nodes to use.
/// * `release` - The compilation mode for the rust compiler.
///
/// # Returns
/// The exit status of the compose script.
fn setup_docker(nodes: usize, release: bool) -> io::Result<ExitStatus> {
    let mut cmd = Command::new("python3");

    cmd.arg(COMPOSE_FILE_PATH)
        .arg("--nodes")
        .arg(nodes.to_string());

    if release {
        cmd.arg("--release");
    }

    cmd.spawn()?.wait()
}

/// Builds the addresses for nodes.
///
/// # Args
/// * `nodes` - The amount of nodes to use.
///
/// # Returns
/// A tuple of lists of addresses.
fn build_addresses(nodes: usize) -> Vec<String> {
    (0..nodes)
        .map(|i| format!("node-{i}:{}", NODE_BASE_PORT + i))
        .collect()
}

fn main() -> io::Result<()> {
    unsafe { env::set_var("RUST_LOG", "debug") };
    env_logger::init();

    const WORKERS: usize = 3;
    const SERVERS: usize = 2;
    const NODES: usize = WORKERS + SERVERS;
    const RELEASE: bool = false;

    setup_docker(NODES, RELEASE)?;

    thread::sleep(Duration::from_secs(2));
    let addrs = build_addresses(NODES);

    let synchronizer_config = SynchronizerConfig::NonBlocking;
    let store_config = StoreConfig::Wild;

    #[allow(unused_variables)]
    let parameter_server_config = AlgorithmConfig::ParameterServer {
        nservers: NonZeroUsize::new(SERVERS).unwrap(),
        synchronizer: synchronizer_config,
        store: store_config,
    };

    #[allow(unused_variables)]
    let all_reduce_config = AlgorithmConfig::AllReduce;

    #[allow(unused_variables)]
    let strategy_switch_config = AlgorithmConfig::StrategySwitch {
        nservers: NonZeroUsize::new(SERVERS).unwrap(),
        synchronizer: synchronizer_config,
        store: store_config,
    };

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

    let training_config = TrainingConfig {
        addrs,
        algorithm: strategy_switch_config,
        serializer: SerializerConfig::SparseCapable {
            r: Float01::new(0.9).unwrap(),
        },
        dataset: DatasetConfig {
            src: DataSrc::Inline {
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
            lr: FloatPositive::new(1.0).unwrap(),
            mu: Float01::new(0.9).unwrap(),
        },
        loss_fn: LossFnConfig::Mse,
        batch_size: NonZeroUsize::new(4).unwrap(),
        max_epochs: NonZeroUsize::new(100).unwrap(),
        offline_epochs: 0,
        seed: Some(42),
        early_stopping: None,
    };

    let session = train(model_config, training_config).unwrap();
    let (_cancel, cancel_rx) = CancelHandle::pair();
    let mut rx = session.event_listener(cancel_rx);

    loop {
        match rx.blocking_recv() {
            Some(orchestrator::TrainingEvent::PublishedLosses { losses, worker_id }) => {
                info!("losses: {worker_id}: {losses:?}");
            }
            Some(orchestrator::TrainingEvent::TrainingComplete {
                model: trained,
                stop_reason: reason,
            }) => {
                info!("params: {:?}", trained.params);
                info!("stop reason: {reason:?}");

                trained
                    .save_safetensors(MODEL_OUTPUT_PATH)
                    .expect("failed to save model");

                info!("saved model to {MODEL_OUTPUT_PATH}");
                break;
            }
            None => break,
            res => info!("result: {res:?}"),
        }
    }

    Ok(())
}
