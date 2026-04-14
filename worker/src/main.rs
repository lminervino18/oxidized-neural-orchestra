use std::{env, io};

use comms::{
    msg::{Command, Msg},
    recv_dataset::{get_dataset_cursor, recv_dataset},
    specs::worker::{AlgorithmSpec, SerializerSpec},
};
use log::{info, warn};
use tokio::{
    net::{TcpListener, TcpStream},
    signal,
};

use worker::{
    builder::WorkerBuilder,
    middlewares::{all_reduce::AllReduceMiddleware, parameter_server::ParameterServerMiddleware},
};

const DEFAULT_HOST: &str = "127.0.0.1";

#[tokio::main]
async fn main() -> io::Result<()> {
    env_logger::init();

    let host = env::var("HOST").unwrap_or_else(|_| DEFAULT_HOST.to_string());
    let port = env::var("PORT").map_err(io::Error::other)?;
    let bind_addr = format!("{host}:{port}");
    let worker_addr = env::var("WORKER_ADDR").unwrap_or_else(|_| bind_addr.clone());

    let list = TcpListener::bind(&bind_addr).await?;
    info!("listening at {bind_addr}");

    let (stream, orch_addr) = list.accept().await?;
    let (rx, tx) = stream.into_split();
    let (mut rx, tx) = comms::channel(rx, tx);
    info!("orchestrator connected from {orch_addr}");

    let spec = loop {
        match rx.recv().await {
            Ok(Msg::Control(Command::CreateWorker(spec))) => break spec,
            Ok(msg) => warn!("expected CreateWorker, got {msg:?}"),
            Err(e) => warn!("io error {e}"),
        }
    };

    // TODO: esto quizás no debería estar acá...
    let x_size_bytes = spec.dataset.x_size_bytes as usize;
    let y_size_bytes = spec.dataset.y_size_bytes as usize;
    let mut samples_raw = vec![0.; x_size_bytes / size_of::<f32>()];
    let mut labels_raw = vec![0.; y_size_bytes / size_of::<f32>()];
    recv_dataset(
        &mut get_dataset_cursor(&mut samples_raw),
        &mut get_dataset_cursor(&mut labels_raw),
        x_size_bytes,
        y_size_bytes,
        &mut rx,
    )
    .await?;

    match spec.algorithm.clone() {
        AlgorithmSpec::ParameterServer {
            server_addrs,
            server_sizes,
            server_ordering,
        } => {
            let serializer = spec.serializer.clone();
            let worker_builder = WorkerBuilder::new();
            let worker =
                worker_builder.build_parameter_server(spec, &server_sizes, samples_raw, labels_raw);
            let mut middleware = ParameterServerMiddleware::new(server_ordering);

            for (addr, size) in server_addrs.into_iter().zip(server_sizes) {
                let stream = TcpStream::connect(addr).await?;
                let (rx, tx) = stream.into_split();

                let (rx, tx) = match serializer {
                    SerializerSpec::Base => comms::channel(rx, tx),
                    SerializerSpec::SparseCapable { r, seed } => {
                        comms::sparse_tx_channel(rx, tx, r, seed)
                    }
                };

                middleware.spawn(rx, tx, size);
            }

            tokio::select! {
                ret = worker.run(rx, tx, middleware) => {
                    ret?;
                    info!("wrapping up, disconnecting...");
                }
                _ = signal::ctrl_c() => {
                    info!("received SIGTERM");
                }
            }
        }
        AlgorithmSpec::AllReduce { worker_addrs } => {
            let worker_builder = WorkerBuilder::new();
            let worker = worker_builder.build_all_reduce(
                spec,
                worker_addr.clone(),
                worker_addrs.clone(),
                samples_raw,
                labels_raw,
            );
            let middleware = AllReduceMiddleware::new(&worker_addr, worker_addrs)?;

            tokio::select! {
                ret = worker.run(rx, tx, middleware) => {
                    ret?;
                    info!("wrapping up, disconnecting...");
                }
                _ = signal::ctrl_c() => {
                    info!("received SIGTERM");
                }
            }
        }
    }

    Ok(())
}
