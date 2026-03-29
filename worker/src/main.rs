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

    let spec = loop {
        match rx.recv().await {
            Ok(Msg::Control(Command::CreateWorker(spec))) => break spec,
            Ok(msg) => warn!("expected CreateWorker, got {msg:?}"),
            Err(e) => warn!("io error {e}"),
        }
    };

    let size = spec.dataset.size;
    let mut dataset_raw = vec![0f32; (size / size_of::<f32>() as u64) as usize];
    recv_dataset(&mut get_dataset_cursor(&mut dataset_raw), size, &mut rx).await?;

    let AlgorithmSpec::ParameterServer {
        server_addrs,
        server_sizes,
        server_ordering,
    } = spec.algorithm.clone();

    let serializer = spec.serializer.clone();
    let worker_builder = WorkerBuilder::new();
    let worker = worker_builder.build(spec, &server_sizes, dataset_raw);
    let mut middleware = Middleware::new(server_ordering);

    for (addr, size) in server_addrs.into_iter().zip(server_sizes) {
        let stream = TcpStream::connect(addr).await?;
        let (rx, tx) = stream.into_split();

        let (rx, tx) = match serializer {
            SerializerSpec::Base => comms::channel(rx, tx),
            SerializerSpec::SparseCapable { r, seed } => comms::sparse_tx_channel(rx, tx, r, seed),
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

    Ok(())
}
