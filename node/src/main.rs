use std::{collections::HashMap, env, io};

use comms::{
    msg::{Command, Msg, Payload},
    recv_dataset::{get_dataset_cursor, recv_dataset},
};
use log::{info, warn};
use parameter_server::service::ServerBuilder;
use tokio::{
    net::{
        TcpListener,
        tcp::{OwnedReadHalf, OwnedWriteHalf},
    },
    signal,
    sync::mpsc,
    task,
};
use worker::builder::WorkerBuilder;

use comms::{OnoReceiver, OnoSender};

const DEFAULT_HOST: &str = "0.0.0.0";

#[tokio::main]
async fn main() -> io::Result<()> {
    env_logger::init();

    let host = env::var("HOST").unwrap_or_else(|_| DEFAULT_HOST.to_string());
    let port = env::var("PORT").map_err(io::Error::other)?;
    let bind_addr = format!("{host}:{port}");

    let listener = TcpListener::bind(&bind_addr).await?;
    info!("listening at {bind_addr}");

    let local = task::LocalSet::new();
    local
        .run_until(async move {
            let mut next_session_id: u64 = 0;
            let mut sessions: HashMap<
                u64,
                (usize, mpsc::Sender<(OwnedReadHalf, OwnedWriteHalf)>),
            > = HashMap::new();

            loop {
                tokio::select! {
                    accept = listener.accept() => {
                        let (stream, addr) = accept?;
                        info!("connection from {addr}");

                        let (rx, tx) = stream.into_split();
                        let (mut rx, mut tx) = comms::channel(rx, tx);

                        let first_msg = match rx.recv().await {
                            Ok(msg) => msg,
                            Err(e) => {
                                warn!("error reading first message from {addr}: {e}");
                                continue;
                            }
                        };

                        match first_msg {
                            Msg::Control(Command::CreateWorker(spec)) => {
                                info!("bootstrapping worker session for {addr}");
                                let bind_addr_clone = bind_addr.clone();
                                task::spawn_local(run_worker(rx, tx, spec, bind_addr_clone));
                            }
                            Msg::Control(Command::CreateServer(spec)) => {
                                info!("bootstrapping server session for {addr}");
                                let session_id = next_session_id;
                                next_session_id += 1;

                                let nworkers = spec.nworkers;
                                let mut pserver = match ServerBuilder::new()
                                    .build(spec)
                                    .map_err(io::Error::other)
                                {
                                    Ok(p) => p,
                                    Err(e) => {
                                        warn!("failed to build server: {e}");
                                        continue;
                                    }
                                };

                                if let Err(e) = tx
                                    .send(&Msg::Control(Command::ServerReady { session_id }))
                                    .await
                                {
                                    warn!("failed to send ServerReady: {e}");
                                    continue;
                                }

                                let (conn_tx, mut conn_rx) =
                                    mpsc::channel::<(OwnedReadHalf, OwnedWriteHalf)>(nworkers);
                                sessions.insert(session_id, (nworkers, conn_tx));

                                tokio::spawn(async move {
                                    for _ in 0..nworkers {
                                        match conn_rx.recv().await {
                                            Some((raw_rx, raw_tx)) => {
                                                pserver.spawn(raw_rx, raw_tx);
                                            }
                                            None => {
                                                warn!(
                                                    "session {session_id}: channel closed before all workers connected"
                                                );
                                                return;
                                            }
                                        }
                                    }
                                    run_server(tx, pserver).await;
                                });
                            }
                            Msg::Control(Command::JoinServer { session_id }) => {
                                info!("routing worker connection to session {session_id}");
                                let mut remove = false;
                                match sessions.get_mut(&session_id) {
                                    None => {
                                        warn!("unknown session id {session_id} from {addr}");
                                    }
                                    Some((remaining, conn_tx)) => {
                                        let raw_rx = rx.into_inner();
                                        let raw_tx = tx.into_inner();
                                        if conn_tx.send((raw_rx, raw_tx)).await.is_ok() {
                                            *remaining -= 1;
                                            if *remaining == 0 {
                                                remove = true;
                                            }
                                        } else {
                                            warn!("session {session_id} channel closed");
                                            remove = true;
                                        }
                                    }
                                }
                                if remove {
                                    sessions.remove(&session_id);
                                }
                            }
                            msg => {
                                warn!("unexpected first message from {addr}: {msg:?}");
                            }
                        }
                    }
                    _ = signal::ctrl_c() => {
                        info!("received SIGTERM");
                        break;
                    }
                }
            }

            Ok(())
        })
        .await
}

async fn run_worker(
    mut rx: OnoReceiver<OwnedReadHalf>,
    tx: OnoSender<OwnedWriteHalf>,
    spec: comms::specs::worker::WorkerSpec,
    bind_addr: String,
) {
    let x_size_bytes = spec.dataset.x_size_bytes as usize;
    let y_size_bytes = spec.dataset.y_size_bytes as usize;
    let mut samples_raw = vec![0f32; x_size_bytes / size_of::<f32>()];
    let mut labels_raw = vec![0f32; y_size_bytes / size_of::<f32>()];

    if let Err(e) = recv_dataset(
        &mut get_dataset_cursor(&mut samples_raw),
        &mut get_dataset_cursor(&mut labels_raw),
        x_size_bytes,
        y_size_bytes,
        &mut rx,
    )
    .await
    {
        warn!("failed to receive dataset: {e}");
        return;
    }

    let worker = match WorkerBuilder::new()
        .build(spec, bind_addr, samples_raw, labels_raw)
        .await
    {
        Ok(w) => w,
        Err(e) => {
            warn!("failed to build worker: {e}");
            return;
        }
    };

    if let Err(e) = worker.run(rx, tx).await {
        warn!("worker session error: {e}");
    } else {
        info!("worker session complete");
    }
}

async fn run_server(
    tx: OnoSender<OwnedWriteHalf>,
    mut pserver: Box<dyn parameter_server::service::Server<OwnedReadHalf, OwnedWriteHalf>>,
) {
    let mut tx = tx;
    match pserver.run().await {
        Ok(mut params) => {
            info!("server session complete, sending parameters");
            let msg = Msg::Data(Payload::Params(&mut params));
            if let Err(e) = tx.send(&msg).await {
                warn!("failed to send parameters: {e}");
            }
        }
        Err(e) => {
            warn!("server session error: {e}");
        }
    }
}
