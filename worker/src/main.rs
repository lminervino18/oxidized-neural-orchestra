use std::{env, error::Error};

use log::info;
use tokio::{net::TcpStream, signal};

use worker::{WorkerAcceptor, WorkerBuilder};

const DEFAULT_HOST: &str = "127.0.0.1";
const DEFAULT_PORT: &str = "8765";

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();

    let addr = format!(
        "{}:{}",
        env::var("HOST").unwrap_or_else(|_| DEFAULT_HOST.to_string()),
        env::var("PORT").unwrap_or_else(|_| DEFAULT_PORT.to_string())
    );

    info!("connecting to server at {addr}");
    let stream = TcpStream::connect(&addr).await?;
    info!("connected to server");

    let (rx, tx) = stream.into_split();
    let (mut rx, tx) = comms::channel(rx, tx);

    let Some(spec) = WorkerAcceptor::handshake(&mut rx).await? else {
        info!("disconnected before bootstrap");
        return Ok(());
    };

    info!(
        "worker bootstrapped: worker_id={}, steps={}, num_params={}, strategy={:?}",
        spec.worker_id,
        spec.steps.get(),
        spec.num_params.get(),
        spec.strategy
    );

    let worker = WorkerBuilder::build(&spec);

    tokio::select! {
        ret = worker.run(rx, tx) => {
            ret?;
            info!("worker finished");
        },
        _ = signal::ctrl_c() => {
            info!("received SIGTERM");
        },
    }

    Ok(())
}
