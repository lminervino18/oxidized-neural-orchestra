use std::{env, error::Error};

use log::info;
use tokio::{net::TcpStream, signal};

use worker::{AlgorithmConnector, WorkerAcceptor, WorkerBuilder, WorkerError};

const DEFAULT_HOST: &str = "127.0.0.1";
const DEFAULT_PORT: &str = "8765";

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();

    let orchestrator_addr = format!(
        "{}:{}",
        env::var("HOST").unwrap_or_else(|_| DEFAULT_HOST.to_string()),
        env::var("PORT").unwrap_or_else(|_| DEFAULT_PORT.to_string())
    );

    info!("connecting to orchestrator at {orchestrator_addr}");
    let stream = TcpStream::connect(&orchestrator_addr).await?;
    info!("connected to orchestrator");

    let (rx, tx) = stream.into_split();
    let (mut rx, _tx) = comms::channel(rx, tx);

    let Some(spec) = WorkerAcceptor::handshake(&mut rx).await? else {
        info!("disconnected before bootstrap");
        return Ok(());
    };

    info!(
        "worker bootstrapped: worker_id={}, steps={}, num_params={}, model={:?}, offline_steps={}, epochs={}",
        spec.worker_id,
        spec.steps.get(),
        spec.num_params.get(),
        spec.model,
        spec.training.offline_steps,
        spec.training.epochs.get(),
    );

    info!("connecting to algorithm data plane: {:?}", spec.training.algorithm);
    let (algo_rx, algo_tx) = AlgorithmConnector::connect(&spec.training.algorithm).await?;
    info!("connected to algorithm data plane");

    let worker = WorkerBuilder::build(&spec);

    tokio::select! {
        ret = worker.run(algo_rx, algo_tx) => {
            ret.map_err(WorkerError::into_io)?;
            info!("worker finished");
        },
        _ = signal::ctrl_c() => {
            info!("received SIGTERM");
        },
    }

    Ok(())
}
