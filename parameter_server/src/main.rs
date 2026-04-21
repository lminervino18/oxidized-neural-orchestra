mod initialization;
mod optimization;
mod service;
mod storage;
mod synchronization;
mod test;

use std::{env, io, time::Duration};

use comms::{Acceptor, Connection, OrchEvent, PullSpecResponse};
use log::{info, warn};
use tokio::{net::TcpListener, signal};

use crate::service::ServerBuilder;

const DEFAULT_HOST: &str = "0.0.0.0";

#[tokio::main]
async fn main() -> io::Result<()> {
    env_logger::init();

    let addr = format!(
        "{}:{}",
        env::var("HOST").unwrap_or_else(|_| DEFAULT_HOST.to_string()),
        env::var("PORT").map_err(io::Error::other)?,
    );

    let listener = TcpListener::bind(&addr).await?;
    info!("listening at {addr}");

    let stream_factory = async move || {
        let (stream, addr) = listener.accept().await?;
        info!("new incoming connection from {addr}");
        Ok(stream.into_split())
    };

    let mut acceptor = Acceptor::new(
        stream_factory,
        Duration::from_secs(5),
        Duration::from_secs(2),
        2,
        5,
    );

    let Connection::Orchestrator(mut orch_handle) = acceptor.accept().await? else {
        panic!("Received an invalid connection type, expected orchestrator");
    };

    let spec = loop {
        match orch_handle.pull_specification().await {
            Ok(PullSpecResponse::ParameterServer(spec)) => break spec,
            Ok(_) => warn!("expected create server, got worker instead"),
            Err(e) => warn!("io error {e}"),
        }
    };

    let nworkers = spec.nworkers;
    let mut pserver = ServerBuilder::new().build(spec).map_err(io::Error::other)?;

    for i in 0..nworkers {
        let Connection::Worker(worker_handle) = acceptor.accept().await? else {
            panic!("Received an invalid connectin type");
        };

        info!("worker {i}/{nworkers} connected");
        pserver.spawn(worker_handle);
    }

    tokio::select! {
        ret = pserver.run() => {
            info!("wrapping up, sending parameters...");
            let mut params = ret?;

            loop {
                match orch_handle.recv_event().await? {
                    OrchEvent::RequestParams => orch_handle.push_params(&mut params).await?,
                    OrchEvent::Disconnect => orch_handle.disconnect().await?,
                }
            }
        },
        _ = signal::ctrl_c() => {
            info!("received SIGTERM");
        },
    }

    Ok(())
}
