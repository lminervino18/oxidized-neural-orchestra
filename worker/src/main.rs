use std::{env, io};

use comms::{
    msg::{Command, Msg},
    recv_dataset::{get_dataset_cursor, recv_dataset},
    specs::{
        algorithm::AlgorithmSpec,
        worker::WorkerSpec,
    },
};
use log::{info, warn};
use tokio::{
    net::{TcpListener, TcpStream},
    signal,
};

use worker::{builder::WorkerBuilder, middleware::Middleware};

const DEFAULT_HOST: &str = "127.0.0.1";

#[tokio::main]
async fn main() -> io::Result<()> {
    env_logger::init();

    let addr = format!(
        "{}:{}",
        env::var("HOST").unwrap_or_else(|_| DEFAULT_HOST.to_string()),
        env::var("PORT").map_err(io::Error::other)?,
    );

    let list = TcpListener::bind(&addr).await?;
    info!("listening at {addr}");

    let (stream, addr) = list.accept().await?;
    let (rx, tx) = stream.into_split();
    let (mut rx, tx) = comms::channel(rx, tx);
    info!("orchestrator connected from {addr}");

    let mut rx_buf = vec![0; 1028];
    let spec = loop {
        match rx.recv_into(&mut rx_buf).await {
            Ok(Msg::Control(Command::CreateWorker(spec))) => break spec,
            Ok(msg) => warn!("expected CreateWorker, got {msg:?}"),
            Err(e) => warn!("io error {e}"),
        }
    };

    let size = spec.dataset.size;
    let mut dataset_raw = vec![0f32; (size / size_of::<f32>() as u64) as usize];
    recv_dataset(&mut get_dataset_cursor(&mut dataset_raw), size, &mut rx).await?;

    tokio::select! {
        ret = run_worker(spec, dataset_raw, rx, tx) => {
            ret?;
            info!("wrapping up, disconnecting...");
        }
        _ = signal::ctrl_c() => {
            info!("received SIGTERM");
        }
    }

    Ok(())
}

/// Runs the worker according to its configured distributed algorithm.
///
/// # Args
/// * `spec` - The worker bootstrap specification received from the orchestrator.
/// * `dataset_raw` - The raw dataset partition assigned to this worker.
/// * `rx` - The receiving end of the communication between the worker and the orchestrator.
/// * `tx` - The sending end of the communication between the worker and the orchestrator.
///
/// # Returns
/// An io error if occurred.
async fn run_worker(
    spec: WorkerSpec,
    dataset_raw: Vec<f32>,
    rx: comms::OnoReceiver<tokio::net::tcp::OwnedReadHalf>,
    tx: comms::OnoSender<tokio::net::tcp::OwnedWriteHalf>,
) -> io::Result<()> {
    match spec.algorithm.clone() {
        AlgorithmSpec::ParameterServer(ps_spec) => {
            run_parameter_server_worker(spec, ps_spec, dataset_raw, rx, tx).await
        }
        AlgorithmSpec::RingAllReduce(_) => Err(io::Error::other(
            "ring all-reduce worker runtime is not implemented yet",
        )),
    }
}

/// Runs the parameter-server worker runtime.
///
/// # Args
/// * `spec` - The worker bootstrap specification received from the orchestrator.
/// * `ps_spec` - The parameter-server algorithm specification.
/// * `dataset_raw` - The raw dataset partition assigned to this worker.
/// * `rx` - The receiving end of the communication between the worker and the orchestrator.
/// * `tx` - The sending end of the communication between the worker and the orchestrator.
///
/// # Returns
/// An io error if occurred.
async fn run_parameter_server_worker(
    spec: WorkerSpec,
    ps_spec: comms::specs::algorithm::ParameterServerSpec,
    dataset_raw: Vec<f32>,
    rx: comms::OnoReceiver<tokio::net::tcp::OwnedReadHalf>,
    tx: comms::OnoSender<tokio::net::tcp::OwnedWriteHalf>,
) -> io::Result<()> {
    let worker_builder = WorkerBuilder::new();
    let worker = worker_builder.build(spec, &ps_spec.server_sizes, dataset_raw);
    let mut middleware = Middleware::new(ps_spec.server_ordering);

    for (addr, size) in ps_spec.server_addrs.into_iter().zip(ps_spec.server_sizes) {
        let stream = TcpStream::connect(addr).await?;
        let (rx, tx) = stream.into_split();
        let (rx, tx) = comms::channel(rx, tx);
        middleware.spawn(rx, tx, size);
    }

    worker.run_parameter_server(rx, tx, middleware).await
}