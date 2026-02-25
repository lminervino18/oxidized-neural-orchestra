use std::{env, io};

use comms::specs::worker::AlgorithmSpec;
use env_logger::Env;
use log::info;
use tokio::{net::TcpListener, signal};

use worker::{WorkerAcceptor, WorkerBuilder, middleware::Middleware};

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
    let (mut rx, mut tx) = comms::channel(rx, tx);
    info!("orchestrator connected from {addr}");

    let mut buf = vec![0; 1028];

    let spec = loop {
        match rx.recv_into(&mut buf).await {
            Ok(Msg::Control(Command::CreateWorker(spec))) => break spec,
            Ok(msg) => warn!("expected CreateWorker, got {msg:?}"),
            Err(e) => warn!("io error {e}"),
        }
    };

    let layer_sizes = spec
    let AlgorithmSpec::ParameterServer {
        server_addrs,
        server_sizes,
        server_ordering,
    } = spec.algorithm
    else {
        unimplemented!();
    };


    let worker_builder = WorkerBuilder::new();
    let worker = worker_builder.build(spec, server_sizes);
    let middleware = Middleware::new(server_ordering, layer_sizes);

    tokio::select! {
        ret = worker.run(rx, tx, server_addrs, server_sizes, server_ordering) => {
            ret?;
            info!("wrapping up, disconnecting...");
        }
        _ = signal::ctrl_c() => {
            info!("received SIGTERM");
        }
    }

    Ok(())
}
