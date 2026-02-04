use std::{env, error::Error};

use env_logger::Env;
use log::info;
use tokio::{net::TcpListener, signal};

use worker::{WorkerAcceptor, WorkerBuilder, WorkerError};

const DEFAULT_HOST: &str = "127.0.0.1";
const DEFAULT_PORT: &str = "8765";

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    init_logging();

    let host = env::var("HOST").unwrap_or_else(|_| DEFAULT_HOST.to_string());
    let port = env::var("PORT").unwrap_or_else(|_| DEFAULT_PORT.to_string());
    let listen_addr = format!("{host}:{port}");

    info!("listening for orchestrator at {listen_addr}");
    let listener = TcpListener::bind(&listen_addr).await?;

    let (stream, peer) = listener.accept().await?;
    info!("orchestrator connected from {peer}");

    let (rx, tx) = stream.into_split();
    let (mut rx, _tx) = comms::channel(rx, tx);

    let Some(spec) = WorkerAcceptor::bootstrap(&mut rx).await? else {
        info!("disconnected before bootstrap");
        return Ok(());
    };

    info!("worker bootstrapped: worker_id={}", spec.worker_id);

    let worker = WorkerBuilder::build(spec);

    tokio::select! {
        ret = worker.run() => {
            ret.map_err(WorkerError::into_io)?;
            info!("worker finished");
        }
        _ = signal::ctrl_c() => {
            info!("received shutdown signal");
        }
    }

    Ok(())
}

fn init_logging() {
    env_logger::Builder::from_env(Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();
}
