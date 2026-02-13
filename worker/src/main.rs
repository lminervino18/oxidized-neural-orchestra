use std::{env, io};

use env_logger::Env;
use log::info;
use tokio::{net::TcpListener, signal};

use worker::{WorkerAcceptor, WorkerBuilder};

const DEFAULT_HOST: &str = "127.0.0.1";

#[tokio::main]
async fn main() -> io::Result<()> {
    init_logging();

    let host = env::var("HOST").unwrap_or_else(|_| DEFAULT_HOST.to_string());
    let port = env::var("PORT").map_err(|e| io::Error::other(e))?;
    let listen_addr = format!("{host}:{port}");

    let (mut orch_rx, orch_tx) = accept_orchestrator_channel(&listen_addr).await?;

    let Some(spec) = WorkerAcceptor::bootstrap(&mut orch_rx).await? else {
        info!("disconnected before bootstrap");
        return Ok(());
    };

    info!("worker bootstrapped: worker_id={}", spec.worker_id);

    let worker = WorkerBuilder::build(spec);

    tokio::select! {
        ret = worker.run(orch_rx, orch_tx) => {
            ret.map_err(io::Error::from)?;
            info!("worker finished");
        }
        _ = signal::ctrl_c() => {
            info!("received shutdown signal");
        }
    }

    Ok(())
}

/// Accepts an orchestrator connection and returns the communication channel.
///
/// # Args
/// * `listen_addr` - Address to bind and accept the orchestrator connection.
///
/// # Returns
/// A receiver/sender pair for orchestrator messages.
///
/// # Errors
/// Returns an `io::Error` if binding or accept fails.
async fn accept_orchestrator_channel(
    listen_addr: &str,
) -> io::Result<(
    comms::OnoReceiver<tokio::net::tcp::OwnedReadHalf>,
    comms::OnoSender<tokio::net::tcp::OwnedWriteHalf>,
)> {
    info!("listening for orchestrator at {listen_addr}");
    let listener = TcpListener::bind(listen_addr).await?;

    let (stream, peer) = listener.accept().await?;
    info!("orchestrator connected from {peer}");

    let (rx, tx) = stream.into_split();
    let (rx, tx) = comms::channel(rx, tx);

    Ok((rx, tx))
}

fn init_logging() {
    env_logger::Builder::from_env(Env::default())
        .format_timestamp_millis()
        .init();
}
