use std::{
    collections::HashMap,
    env, io,
    sync::{Arc, Mutex},
    time::Duration,
};

use comms::{
    Acceptor, Connection, NetRtp, OrchHandle, PullSpecResponse, WorkerEvent, WorkerHandle,
};
use log::{info, warn};
use tokio::{
    net::{
        TcpListener,
        tcp::{OwnedReadHalf, OwnedWriteHalf},
    },
    signal,
    sync::mpsc::{self, Sender},
};

mod router;

use router::{run_server, run_worker};

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

    let acceptor = Acceptor::new(
        stream_factory,
        Duration::from_secs(5),
        Duration::from_secs(2),
        2,
        5,
    );

    NodeRouter::new(acceptor).run().await
}

struct RouterState {
    next_session_id: u64,
    sessions: HashMap<u64, (usize, Sender<WorkerHandle<NetRtp>>)>,
}

impl RouterState {
    fn new() -> Self {
        Self { next_session_id: 0, sessions: HashMap::new() }
    }
}

struct NodeRouter<F>
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
    fn new(acceptor: Acceptor<OwnedReadHalf, OwnedWriteHalf, F>) -> Self {
        Self { acceptor, state: Arc::new(Mutex::new(RouterState::new())) }
    }

    async fn run(&mut self) -> io::Result<()> {
        loop {
            tokio::select! {
                accept = self.acceptor.accept() => {
                    match accept {
                        Ok(conn) => {
                            let state = Arc::clone(&self.state);
                            tokio::spawn(handle_connection(conn, state));
                        }
                        Err(e) => warn!("accept error: {e}"),
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
}

async fn handle_connection(conn: Connection<NetRtp>, state: Arc<Mutex<RouterState>>) {
    match conn {
        Connection::Orchestrator(orch_handle) => bootstrap_node(orch_handle, state).await,
        Connection::Worker(worker_handle) => route_worker(worker_handle, state).await,
        Connection::ParamServer(_) => warn!("unexpected ParamServer connection"),
    }
}

async fn bootstrap_node(mut orch_handle: OrchHandle<NetRtp>, state: Arc<Mutex<RouterState>>) {
    match orch_handle.pull_specification().await {
        Ok(PullSpecResponse::Worker(spec)) => {
            info!("bootstrapping worker session");
            tokio::spawn(run_worker(orch_handle, spec));
        }
        Ok(PullSpecResponse::ParameterServer(spec)) => {
            let session_id = {
                let mut s = state.lock().unwrap();
                let id = s.next_session_id;
                s.next_session_id += 1;
                id
            };
            info!("bootstrapping server session {session_id}");

            if let Err(e) = orch_handle.push_session_ready(session_id).await {
                warn!("session {session_id}: failed to notify orchestrator: {e}");
                return;
            }

            let nworkers = spec.nworkers;
            let (worker_tx, worker_rx) = mpsc::channel(nworkers);
            state.lock().unwrap().sessions.insert(session_id, (nworkers, worker_tx));

            tokio::spawn(run_server(session_id, orch_handle, spec, worker_rx));
        }
        Err(e) => warn!("failed to pull specification: {e}"),
    }
}

async fn route_worker(mut worker_handle: WorkerHandle<NetRtp>, state: Arc<Mutex<RouterState>>) {
    match worker_handle.recv_event().await {
        Ok(WorkerEvent::JoinSession(session_id)) => {
            info!("routing worker to session {session_id}");
            attach_worker(session_id, worker_handle, state).await;
        }
        Ok(event) => warn!("unexpected first event from worker: {event:?}"),
        Err(e) => warn!("error reading worker event: {e}"),
    }
}

async fn attach_worker(
    session_id: u64,
    worker_handle: WorkerHandle<NetRtp>,
    state: Arc<Mutex<RouterState>>,
) {
    let tx = {
        let s = state.lock().unwrap();
        s.sessions.get(&session_id).map(|(_, tx)| tx.clone())
    };

    let Some(tx) = tx else {
        warn!("unknown session_id {session_id}");
        return;
    };

    let sent = tx.send(worker_handle).await.is_ok();

    let remove = {
        let mut s = state.lock().unwrap();
        match s.sessions.get_mut(&session_id) {
            Some((remaining, _)) => {
                if sent {
                    *remaining -= 1;
                    *remaining == 0
                } else {
                    warn!("session {session_id}: channel closed");
                    true
                }
            }
            None => false,
        }
    };

    if remove {
        state.lock().unwrap().sessions.remove(&session_id);
    }
}
