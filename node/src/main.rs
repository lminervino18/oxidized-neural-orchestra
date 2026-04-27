use std::{collections::HashMap, env, io, time::Duration};

use comms::{
    Acceptor, Connection, Connector, NetRtp, OrchEvent, PullSpecResponse, WorkerEvent,
    protocol::Entity,
};
use log::{info, warn};
use tokio::{
    net::TcpListener,
    signal,
    sync::mpsc,
};
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

    let mut next_session_id: u64 = 0;
    let mut sessions: HashMap<u64, (usize, mpsc::Sender<comms::WorkerHandle<NetRtp>>)> =
        HashMap::new();

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

    loop {
        tokio::select! {
            accept = acceptor.accept() => {
                let conn = match accept {
                    Ok(c) => c,
                    Err(e) => { warn!("accept error: {e}"); continue; }
                };

                match conn {
                    Connection::Orchestrator(mut orch_handle) => {
                        match orch_handle.pull_specification().await {
                            Ok(PullSpecResponse::Worker(spec)) => {
                                info!("bootstrapping worker session");
                                tokio::spawn(run_worker(orch_handle, spec));
                            }
                            Ok(PullSpecResponse::ParameterServer(spec)) => {
                                let session_id = next_session_id;
                                next_session_id += 1;
                                info!("bootstrapping server session {session_id}");

                                if let Err(e) = orch_handle.push_session_ready(session_id).await {
                                    warn!("session {session_id}: failed to notify orchestrator: {e}");
                                    continue;
                                }

                                let nworkers = spec.nworkers;
                                let (worker_tx, worker_rx) = mpsc::channel(nworkers);
                                sessions.insert(session_id, (nworkers, worker_tx));

                                tokio::spawn(run_server(session_id, orch_handle, spec, worker_rx));
                            }
                            Err(e) => warn!("failed to pull specification: {e}"),
                        }
                    }
                    Connection::Worker(mut worker_handle) => {
                        match worker_handle.recv_event().await {
                            Ok(WorkerEvent::JoinSession { session_id }) => {
                                info!("routing worker to session {session_id}");
                                let mut remove = false;

                                match sessions.get_mut(&session_id) {
                                    Some((remaining, tx)) => {
                                        if tx.send(worker_handle).await.is_ok() {
                                            *remaining -= 1;
                                            if *remaining == 0 {
                                                remove = true;
                                            }
                                        } else {
                                            warn!("session {session_id}: channel closed");
                                            remove = true;
                                        }
                                    }
                                    None => warn!("unknown session_id {session_id}"),
                                }

                                if remove {
                                    sessions.remove(&session_id);
                                }
                            }
                            Ok(event) => warn!("unexpected first event from worker: {event:?}"),
                            Err(e) => warn!("error reading worker event: {e}"),
                        }
                    }
                    Connection::ParamServer(_) => warn!("unexpected ParamServer connection"),
                }
            }
            _ = signal::ctrl_c() => {
                info!("received SIGTERM, shutting down");
                break;
            }
        }
    }

    Ok(())
}

async fn run_worker(
    mut orch_handle: comms::OrchHandle<NetRtp>,
    spec: comms::specs::worker::WorkerSpec,
) {
    let x_size_bytes = spec.dataset.x_size_bytes as usize;
    let y_size_bytes = spec.dataset.y_size_bytes as usize;
    let mut samples_raw = vec![0f32; x_size_bytes / size_of::<f32>()];
    let mut labels_raw = vec![0f32; y_size_bytes / size_of::<f32>()];

    if let Err(e) = orch_handle
        .pull_dataset(
            &mut comms::get_dataset_cursor(&mut samples_raw),
            &mut comms::get_dataset_cursor(&mut labels_raw),
            x_size_bytes,
            y_size_bytes,
        )
        .await
    {
        warn!("failed to receive dataset: {e}");
        return;
    }

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
    let mut worker = match worker_builder
        .build(spec, connector, orch_handle, samples_raw, labels_raw)
        .await
    {
        Ok(w) => w,
        Err(e) => {
            warn!("failed to build worker: {e}");
            return;
        }
    };

    if let Err(e) = worker.run().await {
        warn!("worker session error: {e}");
    } else {
        info!("worker session complete");
    }
}

async fn run_server(
    session_id: u64,
    mut orch_handle: comms::OrchHandle<NetRtp>,
    spec: comms::specs::server::ServerSpec,
    mut worker_rx: mpsc::Receiver<comms::WorkerHandle<NetRtp>>,
) {
    let nworkers = spec.nworkers;

    let mut pserver = match parameter_server::service::ServerBuilder::new()
        .build::<tokio::net::tcp::OwnedReadHalf, tokio::net::tcp::OwnedWriteHalf>(spec)
        .map_err(io::Error::other)
    {
        Ok(p) => p,
        Err(e) => {
            warn!("session {session_id}: failed to build server: {e}");
            return;
        }
    };

    for i in 0..nworkers {
        match worker_rx.recv().await {
            Some(worker_handle) => {
                info!("session {session_id}: worker {}/{nworkers} connected", i + 1);
                pserver.spawn(worker_handle);
            }
            None => {
                warn!(
                    "session {session_id}: worker channel closed before all workers connected ({i}/{nworkers})"
                );
                return;
            }
        }
    }

    match pserver.run().await {
        Ok(mut params) => {
            info!("session {session_id}: training complete, sending parameters");
            loop {
                match orch_handle.recv_event().await {
                    Ok(OrchEvent::RequestParams) => {
                        if let Err(e) = orch_handle.push_params(&mut params).await {
                            warn!("session {session_id}: failed to push params: {e}");
                            return;
                        }
                    }
                    Ok(OrchEvent::Disconnect) => {
                        let _ = orch_handle.disconnect().await;
                        info!("session {session_id}: disconnected");
                        break;
                    }
                    Ok(event) => warn!("session {session_id}: unexpected orch event: {event:?}"),
                    Err(e) => {
                        warn!("session {session_id}: orch handle error: {e}");
                        return;
                    }
                }
            }
        }
        Err(e) => warn!("session {session_id}: server error: {e}"),
    }
}
