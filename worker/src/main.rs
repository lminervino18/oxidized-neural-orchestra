use std::{env, io, time::Duration};

use comms::{
    Acceptor, Connection, Connector, PullSpecResponse,
    protocol::Entity,
    specs::worker::{AlgorithmSpec, SerializerSpec},
};
use log::{info, warn};
use tokio::{net::TcpListener, signal};

use crate::{builder::WorkerBuilder, cluster_manager::ServerClusterManager};

const DEFAULT_HOST: &str = "127.0.0.1";

#[tokio::main]
async fn main() -> io::Result<()> {
    env_logger::init();

    let host = env::var("HOST").unwrap_or_else(|_| DEFAULT_HOST.to_string());
    let port = env::var("PORT").map_err(io::Error::other)?;
    let addr = format!("{host}:{port}");

    let listener = TcpListener::bind(&addr).await?;
    info!("listening at {addr}");

    let stream_factory = async move || {
        let (stream, addr) = listener.accept().await?;
        info!("new incomming connection from {addr}");
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
            Ok(PullSpecResponse::Worker(spec)) => break spec,
            Ok(_) => warn!("expected CreateWorker, got server instead"),
            Err(e) => warn!("io error {e}"),
        }
    };

    let x_size_bytes = spec.dataset.x_size_bytes as usize;
    let y_size_bytes = spec.dataset.y_size_bytes as usize;
    let mut samples_raw = vec![0.; x_size_bytes / size_of::<f32>()];
    let mut labels_raw = vec![0.; y_size_bytes / size_of::<f32>()];

    orch_handle
        .pull_dataset(
            &mut comms::get_dataset_cursor(&mut samples_raw),
            &mut comms::get_dataset_cursor(&mut labels_raw),
            x_size_bytes,
            y_size_bytes,
        )
        .await?;

    let connector = Connector::new(
        Duration::from_secs(5),
        Duration::from_secs(2),
        2,
        5,
        Entity::Worker { id: spec.worker_id },
    );

    let worker_builder = WorkerBuilder::new();
    let worker = worker_builder.build(spec, &server_sizes, samples_raw, labels_raw);
    let mut cluster_manager = ServerClusterManager::new(server_ordering);

    let worker = worker_builder
        .build(spec, bind_addr.clone(), samples_raw, labels_raw)
        .await?;

    tokio::select! {
        ret = worker.run(orch_handle, cluster_manager) => {
            ret?;
            info!("wrapping up, disconnecting...");
        }
        _ = signal::ctrl_c() => {
            info!("received SIGTERM");
        }
    }

    Ok(())
}
