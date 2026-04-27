use std::{env, io, time::Duration};

use comms::{Acceptor, Connection, Connector, OrchEvent, PullSpecResponse, protocol::Entity};
use log::{info, warn};
use parameter_server::service::ServerBuilder;
use tokio::{net::TcpListener, signal};
use worker::builder::WorkerBuilder;

const DEFAULT_HOST: &str = "0.0.0.0";

#[tokio::main]
async fn main() -> io::Result<()> {
    env_logger::init();

    let host = env::var("HOST").unwrap_or_else(|_| DEFAULT_HOST.to_string());
    let port = env::var("PORT").map_err(io::Error::other)?;
    let addr = format!("{host}:{port}");

    let listener = TcpListener::bind(&addr).await?;
    info!("listening at {addr}");

    let stream_factory = async move || {
        let (stream, peer_addr) = listener.accept().await?;
        info!("new incoming connection from {peer_addr}");
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
        return Err(io::Error::other("expected orchestrator connection"));
    };

    match orch_handle.pull_specification().await? {
        PullSpecResponse::Worker(spec) => {
            info!("bootstrapping as worker");

            let x_size_bytes = spec.dataset.x_size_bytes as usize;
            let y_size_bytes = spec.dataset.y_size_bytes as usize;
            let mut samples_raw = vec![0f32; x_size_bytes / size_of::<f32>()];
            let mut labels_raw = vec![0f32; y_size_bytes / size_of::<f32>()];

            orch_handle
                .pull_dataset(
                    &mut comms::get_dataset_cursor(&mut samples_raw),
                    &mut comms::get_dataset_cursor(&mut labels_raw),
                    x_size_bytes,
                    y_size_bytes,
                )
                .await?;

            let transport_factory = |rx, tx| {
                comms::build_reliable_transport(
                    rx,
                    tx,
                    Duration::from_secs(5),
                    Duration::from_secs(2),
                    2,
                    5,
                )
            };

            let connector = Connector::new(
                transport_factory,
                Entity::Worker { id: spec.worker_id },
            );

            let worker_builder = WorkerBuilder::new();
            let mut worker = worker_builder
                .build(spec, connector, orch_handle, samples_raw, labels_raw)
                .await?;

            tokio::select! {
                ret = worker.run() => {
                    ret?;
                    info!("worker done");
                }
                _ = signal::ctrl_c() => {
                    info!("received SIGTERM");
                }
            }
        }
        PullSpecResponse::ParameterServer(spec) => {
            info!("bootstrapping as parameter server");

            let nworkers = spec.nworkers;
            let mut pserver = ServerBuilder::new().build(spec).map_err(io::Error::other)?;

            for i in 0..nworkers {
                let Connection::Worker(worker_handle) = acceptor.accept().await? else {
                    return Err(io::Error::other("expected worker connection"));
                };
                info!("worker {}/{nworkers} connected", i + 1);
                pserver.spawn(worker_handle);
            }

            tokio::select! {
                ret = pserver.run() => {
                    info!("server done, sending parameters");
                    let mut params = ret?;

                    loop {
                        match orch_handle.recv_event().await? {
                            OrchEvent::RequestParams => orch_handle.push_params(&mut params).await?,
                            OrchEvent::Disconnect => {
                                orch_handle.disconnect().await?;
                                break;
                            }
                            event => warn!("unexpected orch event: {event:?}"),
                        }
                    }
                }
                _ = signal::ctrl_c() => {
                    info!("received SIGTERM");
                }
            }
        }
    }

    Ok(())
}
