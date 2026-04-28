use std::{io, time::Duration};

use comms::{
    Connector, NetRtp, OrchEvent, OrchHandle, WorkerHandle,
    get_dataset_cursor, build_reliable_transport,
    protocol::Entity,
    specs::{server::ServerSpec, worker::WorkerSpec},
};
use log::{info, warn};
use parameter_server::service::ServerBuilder;
use tokio::{
    net::tcp::{OwnedReadHalf, OwnedWriteHalf},
    sync::mpsc::Receiver,
};
use worker::builder::WorkerBuilder;

pub async fn run_worker(mut orch_handle: OrchHandle<NetRtp>, spec: WorkerSpec) {
    let x_size_bytes = spec.dataset.x_size_bytes as usize;
    let y_size_bytes = spec.dataset.y_size_bytes as usize;
    let mut samples_raw = vec![0f32; x_size_bytes / size_of::<f32>()];
    let mut labels_raw = vec![0f32; y_size_bytes / size_of::<f32>()];

    if let Err(e) = orch_handle
        .pull_dataset(
            &mut get_dataset_cursor(&mut samples_raw),
            &mut get_dataset_cursor(&mut labels_raw),
            x_size_bytes,
            y_size_bytes,
        )
        .await
    {
        warn!("failed to receive dataset: {e}");
        return;
    }

    let connector = Connector::new(
        |rx, tx| build_reliable_transport(rx, tx, Duration::from_secs(5), Duration::from_secs(2), 2, 5),
        Entity::Worker { id: spec.worker_id },
    );

    let mut worker = match WorkerBuilder::new()
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

pub async fn run_server(
    session_id: u64,
    mut orch_handle: OrchHandle<NetRtp>,
    spec: ServerSpec,
    mut worker_rx: Receiver<WorkerHandle<NetRtp>>,
) {
    let nworkers = spec.nworkers;

    let mut pserver = match ServerBuilder::new()
        .build::<OwnedReadHalf, OwnedWriteHalf>(spec)
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
        Ok(mut params) => serve_params(session_id, &mut orch_handle, &mut params).await,
        Err(e) => warn!("session {session_id}: server error: {e}"),
    }
}

async fn serve_params(session_id: u64, orch_handle: &mut OrchHandle<NetRtp>, params: &mut Vec<f32>) {
    info!("session {session_id}: training complete, sending parameters");
    loop {
        match orch_handle.recv_event().await {
            Ok(OrchEvent::RequestParams) => {
                if let Err(e) = orch_handle.push_params(params).await {
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
