use std::{
    collections::HashMap,
    io,
    sync::{Arc, Mutex},
    time::Duration,
};

use comms::{
    Acceptor, Connection, Connector, NetRtp, OrchEvent, OrchHandle, PullSpecResponse, WorkerEvent,
    WorkerHandle, build_reliable_transport, get_dataset_cursor,
    protocol::Entity,
    specs::{server::ServerSpec, worker::WorkerSpec},
};
use log::{info, warn};
use parameter_server::service::ServerBuilder;
use tokio::{
    net::tcp::{OwnedReadHalf, OwnedWriteHalf},
    sync::mpsc::{self, Receiver, Sender},
};
use worker::builder::WorkerBuilder;

struct RouterState {
    next_session_id: u64,
    sessions: HashMap<u64, (usize, Sender<WorkerHandle<NetRtp>>)>,
}

impl RouterState {
    fn new() -> Self {
        Self {
            next_session_id: 0,
            sessions: HashMap::new(),
        }
    }

    fn register_session(&mut self, nworkers: usize) -> (u64, Receiver<WorkerHandle<NetRtp>>) {
        let session_id = self.next_session_id;
        self.next_session_id += 1;
        let (tx, rx) = mpsc::channel(nworkers);
        self.sessions.insert(session_id, (nworkers, tx));
        (session_id, rx)
    }

    fn take_worker_slot(&mut self, session_id: u64) -> Option<Sender<WorkerHandle<NetRtp>>> {
        let (remaining, tx) = self.sessions.get_mut(&session_id)?;
        *remaining -= 1;
        let tx = tx.clone();
        if *remaining == 0 {
            self.sessions.remove(&session_id);
        }
        Some(tx)
    }
}

pub struct NodeRouter<F>
where
    F: AsyncFnMut() -> io::Result<(OwnedReadHalf, OwnedWriteHalf)>,
{
    acceptor: Acceptor<OwnedReadHalf, OwnedWriteHalf, F>,
    state: Arc<Mutex<RouterState>>,
}

impl<F> NodeRouter<F>
where
    F: AsyncFnMut() -> io::Result<(OwnedReadHalf, OwnedWriteHalf)>,
{
    pub fn new(acceptor: Acceptor<OwnedReadHalf, OwnedWriteHalf, F>) -> Self {
        Self {
            acceptor,
            state: Arc::new(Mutex::new(RouterState::new())),
        }
    }

    pub async fn run(mut self) -> io::Result<()> {
        loop {
            let conn = self.acceptor.accept().await?;
            let state = Arc::clone(&self.state);
            tokio::spawn(handle_connection(conn, state));
        }
    }
}

async fn handle_connection(conn: Connection<NetRtp>, state: Arc<Mutex<RouterState>>) {
    match conn {
        Connection::Orchestrator(orch_handle) => bootstrap_node(orch_handle, state).await,
        Connection::Worker(worker_handle) => route_worker(worker_handle, state).await,
        Connection::ParamServer(_) => warn!("unexpected ParamServer connection on node listener"),
    }
}

async fn bootstrap_node(mut orch_handle: OrchHandle<NetRtp>, state: Arc<Mutex<RouterState>>) {
    match orch_handle.pull_specification().await {
        Ok(PullSpecResponse::ParameterServer(spec)) => {
            let nworkers = spec.nworkers;
            let (session_id, worker_rx) = state.lock().unwrap().register_session(nworkers);

            if let Err(e) = orch_handle.push_session_ready(session_id).await {
                warn!("session {session_id}: failed to send SessionReady: {e}");
                return;
            }

            info!("session {session_id}: server started, expecting {nworkers} worker(s)");
            tokio::spawn(run_server(session_id, orch_handle, spec, worker_rx));
        }
        Ok(PullSpecResponse::Worker(spec)) => {
            info!("worker {}: session started", spec.worker_id);
            tokio::spawn(run_worker(orch_handle, spec));
        }
        Err(e) => warn!("failed to pull specification: {e}"),
    }
}

async fn route_worker(mut worker_handle: WorkerHandle<NetRtp>, state: Arc<Mutex<RouterState>>) {
    let session_id = match worker_handle.recv_event().await {
        Ok(WorkerEvent::JoinSession(id)) => id,
        Ok(event) => {
            warn!("expected JoinSession from worker, got: {event:?}");
            return;
        }
        Err(e) => {
            warn!("failed to receive JoinSession: {e}");
            return;
        }
    };

    let Some(tx) = state.lock().unwrap().take_worker_slot(session_id) else {
        warn!("session {session_id}: no active session found for incoming worker");
        return;
    };

    if tx.send(worker_handle).await.is_err() {
        warn!("session {session_id}: server task dropped before worker could attach");
    }
}

async fn run_worker(mut orch_handle: OrchHandle<NetRtp>, spec: WorkerSpec) {
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

async fn run_server(
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
