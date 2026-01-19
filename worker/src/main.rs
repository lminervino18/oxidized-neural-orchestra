use std::{env, error::Error};

use log::info;
use tokio::{net::TcpStream, signal};

use ml_core::{MlError, StepStats, TrainStrategy};

const DEFAULT_HOST: &str = "127.0.0.1";
const DEFAULT_PORT: &str = "8765";


/// Placeholder strategy used while the real model/plugin factory is not wired yet.
struct NoopStrategy;

impl TrainStrategy for NoopStrategy {
    fn step(&mut self, _weights: &[f32], _grads: &mut [f32]) -> Result<StepStats, MlError> {
        Ok(StepStats::new(1, 0))
    }
}

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
    let (rx, tx) = comms::channel(rx, tx);

    tokio::select! {
        ret = worker::run_bootstrapped(rx, tx, NoopStrategy) => {
            ret?;
            info!("worker finished");
        },
        _ = signal::ctrl_c() => {
            info!("received SIGTERM");
        },
    }

    Ok(())
}
