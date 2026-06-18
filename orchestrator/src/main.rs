use std::{
    env, io,
    num::NonZeroUsize,
    process::{Command, ExitStatus},
    thread,
    time::{Duration, Instant},
};

use comms::floats::{Float01, FloatNonNegative, FloatPositive};
use log::info;
use orchestrator::{CancelHandle, TrainingEvent, configs::*, train};

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

fn nonzero(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).unwrap()
}

fn make_nielsen_mnist_model() -> ModelConfig {
    use ActFnConfig::*;
    use LayerConfig::*;
    use ParamGenConfig::*;

    let layers = vec![
        Conv {
            input_dim: (nonzero(1), nonzero(28), nonzero(28)),
            kernel_dim: (nonzero(10), nonzero(1), nonzero(5)),
            stride: nonzero(1),
            padding: 0,
            init: Kaiming,
            act_fn: None,
        },
        // MaxPooling {
        //     input_dim: (nonzero(10), nonzero(24), nonzero(24)),
        //     filter_size: nonzero(2),
        //     stride: nonzero(2),
        //     padding: 0,
        //     act_fn: None,
        // },
        Dense {
            output_size: nonzero(100),
            init: Kaiming,
            act_fn: Some(ReLU {
                slope: Float01::new(0.0).unwrap(),
            }),
        },
        Dense {
            output_size: nonzero(10),
            init: Kaiming,
            act_fn: Some(Softmax),
        },
    ];

    ModelConfig { layers }
}

fn make_mnist_dataset() -> DatasetConfig {
    let x_size = nonzero(28 * 28);
    let y_size = nonzero(10);

    let src = DataSrc::Local {
        samples_path: "datasets/mnist/mnist_train_samples.bin".into(),
        labels_path: "datasets/mnist/mnist_train_labels.bin".into(),
    };

    DatasetConfig {
        src,
        x_size,
        y_size,
    }
}

fn main() -> io::Result<()> {
    unsafe { env::set_var("RUST_LOG", "debug") };
    env_logger::init();

    const WORKERS: usize = 2;
    const SERVERS: usize = 2;
    const NODES: usize = WORKERS + SERVERS;
    const RELEASE: bool = false;

    setup_docker(NODES, RELEASE)?;

    thread::sleep(Duration::from_secs(4));
    let addrs = build_addresses(NODES);

    #[allow(unused_variables)]
    let parameter_server_config = AlgorithmConfig::ParameterServer {
        nservers: NonZeroUsize::new(SERVERS).unwrap(),
        synchronizer: SynchronizerConfig::NonBlocking,
        store: StoreConfig::Wild,
    };

    #[allow(unused_variables)]
    let all_reduce_config = AlgorithmConfig::AllReduce;

    #[allow(unused_variables)]
    let strategy_switch_config = AlgorithmConfig::StrategySwitch {
        nservers: NonZeroUsize::new(SERVERS).unwrap(),
        synchronizer: SynchronizerConfig::Barrier,
        store: StoreConfig::Blocking,
    };

    let model_config = make_nielsen_mnist_model();

    let training_config = TrainingConfig {
        addrs,
        algorithm: parameter_server_config,
        serializer: SerializerConfig::SparseCapable {
            r: Float01::new(0.9).unwrap(),
        },
        dataset: make_mnist_dataset(),
        optimizer: OptimizerConfig::GradientDescentWithMomentum {
            lr: FloatPositive::new(0.1).unwrap(),
            mu: Float01::new(0.95).unwrap(),
        },
        loss_fn: LossFnConfig::CrossEntropy,
        batch_size: NonZeroUsize::new(10).unwrap(),
        // max_epochs: NonZeroUsize::new(60).unwrap(),
        max_epochs: NonZeroUsize::new(2).unwrap(),
        offline_epochs: 0,
        seed: Some(42),
        early_stopping: Some(EarlyStoppingConfig {
            tolerance: FloatNonNegative::new(0.02).unwrap(),
        }),
    };

    let start = Instant::now();
    let session = train(model_config, training_config).unwrap();
    let (_cancel, cancel_rx) = CancelHandle::pair();
    let mut rx = session.event_listener(cancel_rx);

    loop {
        match rx.blocking_recv() {
            Some(TrainingEvent::PublishedLosses { losses, worker_id }) => {
                info!("losses: {worker_id}: {losses:?}");
            }
            Some(TrainingEvent::TrainingComplete {
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

    println!("took: {} seconds", start.elapsed().as_secs_f32());
    Ok(())
}
