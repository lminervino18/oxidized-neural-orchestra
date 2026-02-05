mod initialization;
mod optimization;
mod service;
mod storage;
mod synchronization;

use std::{env, error::Error};

use comms::msg::{Command, Msg, Payload};
use log::{info, warn};
use tokio::{net::TcpListener, signal};

use crate::service::ServerBuilder;

const DEFAULT_HOST: &str = "0.0.0.0";
const DEFAULT_PORT: &str = "8765";

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();

    let addr = format!(
        "{}:{}",
        env::var("HOST").unwrap_or_else(|_| DEFAULT_HOST.to_string()),
        env::var("PORT").unwrap_or_else(|_| DEFAULT_PORT.to_string())
    );

    let list = TcpListener::bind(&addr).await?;
    info!("listening at {addr}");

    let (stream, addr) = list.accept().await?;
    info!("orchestrator connected from {addr}");
    let (rx, tx) = stream.into_split();
    let (mut rx, mut tx) = comms::channel(rx, tx);

    let spec = loop {
        match rx.recv().await {
            Ok(Msg::Control(Command::CreateServer(spec))) => break spec,
            Ok(msg) => warn!("expected CreateServer, got {msg:?}"),
            Err(e) => warn!("io error {e}"),
        }
    };

    let nworkers = spec.nworkers;
    let mut pserver = ServerBuilder::new().build(spec)?;

    for i in 0..nworkers {
        let (stream, addr) = list.accept().await?;
        info!("worker {i}/{nworkers} connected from {addr}");
        let (rx, tx) = stream.into_split();
        let (rx, tx) = comms::channel(rx, tx);
        pserver.spawn(rx, tx);
    }

    tokio::select! {
        ret = pserver.run() => {
            info!("wrapping up, sending parameters...");
            let mut params = ret?;
            let msg = Msg::Data(Payload::Params(&mut params));
            tx.send(&msg).await?;
        },
        _ = signal::ctrl_c() => {
            info!("received SIGTERM");
        },
    }

    Ok(())
}
