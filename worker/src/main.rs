use std::{
    env,
    io::{self, Cursor},
    num::NonZeroUsize,
};

use comms::{
    msg::{Command, Msg},
    recv_dataset::recv_dataset,
    specs::{machine_learning::DatasetSpec, worker::AlgorithmSpec},
};
use log::{info, warn};
use tokio::{
    net::{TcpListener, TcpStream},
    signal,
};

use machine_learning::dataset::{Dataset, DatasetSrc};
use worker::{builder::WorkerBuilder, middleware::Middleware};

const DEFAULT_HOST: &str = "127.0.0.1";

#[tokio::main]
async fn main() -> io::Result<()> {
    env_logger::init();

    let addr = format!(
        "{}:{}",
        env::var("HOST").unwrap_or_else(|_| DEFAULT_HOST.to_string()),
        env::var("PORT").map_err(|e| io::Error::other(e))?,
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

    let DatasetSpec {
        size,
        x_size,
        y_size,
    } = spec.dataset;

    let mut dataset_raw = vec![0f32; (size / 4) as usize];
    let dataset_bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut dataset_raw);
    recv_dataset(&mut Cursor::new(dataset_bytes), size, &mut rx).await?;

    let dataset_src = DatasetSrc::inline(dataset_raw);
    let dataset = Dataset::new(
        dataset_src,
        NonZeroUsize::new(x_size).unwrap(),
        NonZeroUsize::new(y_size).unwrap(),
    );

    let AlgorithmSpec::ParameterServer {
        server_addrs,
        server_sizes,
        server_ordering,
    } = spec.algorithm.clone();

    let worker_builder = WorkerBuilder::new();
    let worker = worker_builder.build(spec, &server_sizes, dataset);
    let mut middleware = Middleware::new(server_ordering);

    for (addr, size) in server_addrs.into_iter().zip(server_sizes) {
        let stream = TcpStream::connect(addr).await?;
        let (rx, tx) = stream.into_split();
        let (rx, tx) = comms::channel(rx, tx);
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
