use std::{env, error::Error};

use log::info;
use tokio::{net::TcpListener, net::TcpStream, signal};

use comms::specs::worker::AlgorithmSpec;
use worker::{WorkerAcceptor, WorkerBuilder, WorkerError};

const DEFAULT_HOST: &str = "127.0.0.1";
const DEFAULT_PORT: &str = "8765";

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();

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

    info!("worker bootstrapped: {spec:#?}");

    let ps_addr = match spec.training.algorithm {
        AlgorithmSpec::ParameterServer { server_ip } => server_ip,
    };

    info!("connecting to parameter server at {ps_addr}");
    let ps_stream = TcpStream::connect(ps_addr).await?;
    let (ps_rx, ps_tx) = ps_stream.into_split();
    let (ps_rx, ps_tx) = comms::channel(ps_rx, ps_tx);

    let worker = WorkerBuilder::build(&spec);

    tokio::select! {
        ret = worker.run(ps_rx, ps_tx) => {
            ret.map_err(WorkerError::into_io)?;
            info!("worker finished");
        },
        _ = signal::ctrl_c() => {
            info!("received SIGTERM");
        },
    }

    Ok(())
}
