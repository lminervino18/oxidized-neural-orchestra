use std::{env, io};

use comms::{
    msg::{Command, Msg},
    recv_dataset::{get_dataset_cursor, recv_dataset},
    specs::{
        algorithm::{AlgorithmSpec, ParameterServerSpec, RingAllReduceSpec},
        worker::WorkerSpec,
    },
};
use log::{info, warn};
use tokio::{
    net::{TcpListener, TcpStream},
    signal,
};

use worker::{
    builder::WorkerBuilder,
    middleware::{ps::ParameterServerMiddleware, ring::RingMiddleware},
};

const DEFAULT_HOST: &str = "127.0.0.1";

#[tokio::main]
async fn main() -> io::Result<()> {
    env_logger::init();

    let addr = format!(
        "{}:{}",
        env::var("HOST").unwrap_or_else(|_| DEFAULT_HOST.to_string()),
        env::var("PORT").map_err(io::Error::other)?,
    );

    let listener = TcpListener::bind(&addr).await?;
    info!("listening at {addr}");

    let (stream, addr) = listener.accept().await?;
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
        ret = run_worker(spec, dataset_raw, rx, tx, listener) => {
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
/// * `listener` - The worker listener used to accept ring-neighbor connections.
///
/// # Returns
/// An io error if occurred.
async fn run_worker(
    spec: WorkerSpec,
    dataset_raw: Vec<f32>,
    rx: comms::OnoReceiver<tokio::net::tcp::OwnedReadHalf>,
    tx: comms::OnoSender<tokio::net::tcp::OwnedWriteHalf>,
    listener: TcpListener,
) -> io::Result<()> {
    match spec.algorithm.clone() {
        AlgorithmSpec::ParameterServer(ps_spec) => {
            run_parameter_server_worker(spec, ps_spec, dataset_raw, rx, tx).await
        }
        AlgorithmSpec::RingAllReduce(ring_spec) => {
            run_ring_all_reduce_worker(spec, ring_spec, dataset_raw, rx, tx, listener).await
        }
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
    ps_spec: ParameterServerSpec,
    dataset_raw: Vec<f32>,
    rx: comms::OnoReceiver<tokio::net::tcp::OwnedReadHalf>,
    tx: comms::OnoSender<tokio::net::tcp::OwnedWriteHalf>,
) -> io::Result<()> {
    let worker_builder = WorkerBuilder::new();
    let worker = worker_builder.build_parameter_server(spec, &ps_spec.server_sizes, dataset_raw);
    let mut middleware = ParameterServerMiddleware::new(ps_spec.server_ordering);

    for (addr, size) in ps_spec.server_addrs.into_iter().zip(ps_spec.server_sizes) {
        let stream = TcpStream::connect(addr).await?;
        let (rx, tx) = stream.into_split();
        let (rx, tx) = comms::channel(rx, tx);
        middleware.spawn(rx, tx, size);
    }

    worker.run_parameter_server(rx, tx, middleware).await
}

/// Runs the ring all-reduce worker runtime bootstrap.
///
/// # Args
/// * `spec` - The worker bootstrap specification received from the orchestrator.
/// * `ring_spec` - The ring all-reduce algorithm specification.
/// * `dataset_raw` - The raw dataset partition assigned to this worker.
/// * `rx` - The receiving end of the communication between the worker and the orchestrator.
/// * `tx` - The sending end of the communication between the worker and the orchestrator.
/// * `listener` - The worker listener used to accept ring-neighbor connections.
///
/// # Returns
/// An io error if occurred.
async fn run_ring_all_reduce_worker(
    spec: WorkerSpec,
    ring_spec: RingAllReduceSpec,
    dataset_raw: Vec<f32>,
    rx: comms::OnoReceiver<tokio::net::tcp::OwnedReadHalf>,
    tx: comms::OnoSender<tokio::net::tcp::OwnedWriteHalf>,
    listener: TcpListener,
) -> io::Result<()> {
    if ring_spec.worker_addrs.len() < 2 {
        return Err(io::Error::other(
            "ring all-reduce requires at least two workers",
        ));
    }

    let (prev_worker_id, next_worker_id, next_addr) = resolve_ring_neighbors(
        spec.worker_id,
        &ring_spec.ring_ordering,
        &ring_spec.worker_addrs,
    )?;

    let worker_builder = WorkerBuilder::new();
    let worker = worker_builder.build_ring_all_reduce(spec, dataset_raw);

    let next_stream = TcpStream::connect(next_addr).await?;
    let (next_rx, next_tx) = next_stream.into_split();
    let (next_rx, next_tx) = comms::channel(next_rx, next_tx);

    let (prev_stream, _) = listener.accept().await?;
    let (prev_rx, prev_tx) = prev_stream.into_split();
    let (prev_rx, prev_tx) = comms::channel(prev_rx, prev_tx);

    let middleware = RingMiddleware::new(
        prev_worker_id,
        prev_rx,
        prev_tx,
        next_worker_id,
        next_rx,
        next_tx,
    );

    worker.run_ring_all_reduce(rx, tx, middleware).await
}

fn resolve_ring_neighbors<'a>(
    worker_id: usize,
    ring_ordering: &[usize],
    worker_addrs: &'a [String],
) -> io::Result<(usize, usize, &'a str)> {
    let ring_len = ring_ordering.len();
    let rank = ring_ordering
        .iter()
        .position(|&id| id == worker_id)
        .ok_or_else(|| io::Error::other(format!("worker {worker_id} is not part of the ring")))?;

    let prev_rank = if rank == 0 { ring_len - 1 } else { rank - 1 };
    let next_rank = (rank + 1) % ring_len;

    let prev_worker_id = ring_ordering[prev_rank];
    let next_worker_id = ring_ordering[next_rank];
    let next_addr = worker_addrs
        .get(next_worker_id)
        .ok_or_else(|| io::Error::other(format!("missing address for worker {next_worker_id}")))?;

    Ok((prev_worker_id, next_worker_id, next_addr))
}