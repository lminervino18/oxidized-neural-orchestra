use std::{io, thread};
use std::{net::SocketAddr, path::Path};
use tokio::fs::File;

use comms::{
    OnoReceiver, OnoSender,
    msg::{Command, Msg, Payload},
    send_dataset::send_dataset,
    specs::{server::ServerSpec, worker::WorkerSpec},
};
use futures::future;
use log::{debug, error, info, warn};
use tokio::{
    net::{
        TcpStream,
        tcp::{OwnedReadHalf, OwnedWriteHalf},
    },
    runtime::Runtime,
    sync::mpsc::{self, Receiver, Sender},
};

use crate::{OrchErr, Result};

type NetRx = OnoReceiver<OwnedReadHalf>;
type NetTx = OnoSender<OwnedWriteHalf>;

/// An event produced during a training session.
#[derive(Debug)]
pub enum TrainingEvent {
    /// A worker completed an epoch and reported its losses.
    Loss { worker_id: usize, losses: Vec<f32> },
    /// A worker finished and disconnected.
    WorkerDone(usize),
    /// Training completed and all servers returned their final model parameters.
    Complete(Vec<f32>),
    /// A worker or server produced an unrecoverable error.
    Error(OrchErr),
}

/// Represents an ongoing training session.
pub struct Session {
    runtime: Runtime,
    servers: Vec<(NetRx, NetTx)>,
    workers: Vec<(NetRx, NetTx)>,
}

impl Session {
    /// Creates a new session by connecting to all workers and servers.
    ///
    /// # Args
    /// * `workers` - List of (address, spec) pairs for each worker.
    /// * `servers` - List of (address, spec) pairs for each parameter server.
    ///
    /// # Returns
    /// A ready session with all connections established.
    ///
    /// # Errors
    /// Returns an `OrchErr` if any connection or bootstrap message fails.
    pub fn new(
        workers: Vec<(SocketAddr, WorkerSpec)>,
        servers: Vec<(SocketAddr, ServerSpec)>,
    ) -> Result<Self> {
        let (nworkers, nservers) = (workers.len(), servers.len());
        info!("connecting to {nworkers} workers and {nservers} servers",);

        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;

        let server_chans = runtime.block_on(Self::create_servers(servers))?;
        debug!("successfully created all servers");

        let worker_chans = runtime.block_on(Self::create_workers(workers))?;
        debug!("successfully created all workers");

        let session = Self {
            runtime,
            servers: server_chans,
            workers: worker_chans,
        };

        Ok(session)
    }

    async fn worker_listener(id: usize, mut rx: NetRx, _tx: NetTx, tx: Sender<TrainingEvent>) {
        let mut rx_buf = vec![0; 128];

        loop {
            match rx.recv_into(&mut rx_buf).await {
                Ok(Msg::Control(Command::ReportLoss { losses })) => {
                    debug!("worker {id} reported {} losses", losses.len());

                    let event = TrainingEvent::Loss {
                        worker_id: id,
                        losses: losses.into_owned(),
                    };

                    let _ = tx.send(event).await;
                }
                Ok(Msg::Control(Command::Disconnect)) => {
                    info!("worker {id} disconnected");

                    let event = TrainingEvent::WorkerDone(id);
                    let _ = tx.send(event).await;
                    return;
                }
                Ok(msg) => {
                    warn!("worker {id}: unexpected message {msg:?}");

                    let event = TrainingEvent::Error(OrchErr::WorkerError {
                        worker_id: id,
                        msg: format!("unexpected message {msg:?}"),
                    });

                    let _ = tx.send(event).await;
                    return;
                }
                Err(e) if is_eof(&e) => {
                    info!("worker {id} closed connection");

                    let event = TrainingEvent::WorkerDone(id);
                    let _ = tx.send(event).await;
                    return;
                }
                Err(e) => {
                    error!("worker {id} error: {e}");

                    let event = TrainingEvent::Error(OrchErr::WorkerError {
                        worker_id: id,
                        msg: e.to_string(),
                    });

                    let _ = tx.send(event).await;
                    return;
                }
            }
        }
    }

    /// Consumes `self` and creates an event listener for this training session.
    ///
    /// Spawns a background thread that drives the session, listening to all
    /// workers and parameter servers, and forwards events through the channel.
    ///
    /// # Returns
    /// A receiver that yields `TrainingEvent`s as training progresses.
    pub fn event_listener(self) -> Receiver<TrainingEvent> {
        let (tx, rx) = mpsc::channel(256);

        thread::spawn(move || {
            self.runtime.block_on(async move {
                let futs = self.workers.into_iter().enumerate().map(|(i, (wrx, wtx))| {
                    let list = Self::worker_listener(i, wrx, wtx, tx.clone());
                    tokio::spawn(list)
                });

                future::join_all(futs).await;

                // TODO: Instead of using logs, it'd be nice to use events
                debug!("all workers done, reading final params from all servers");

                let server_rxs = self.servers.into_iter().map(|(rx, _)| rx);
                let mut model_params: Vec<f32> = Vec::new();
                let mut rx_buf = vec![0; 1024];

                for (i, mut srx) in server_rxs.into_iter().enumerate() {
                    match srx.recv_into(&mut rx_buf).await {
                        Ok(Msg::Data(Payload::Params(params))) => {
                            model_params.extend_from_slice(params);
                        }
                        Ok(msg) => {
                            let text = format!("unexpected message from server {i}: {msg:?}");
                            let err = OrchErr::ServerError(text);
                            let event = TrainingEvent::Error(err);
                            let _ = tx.send(event).await;
                            return;
                        }
                        Err(e) => {
                            let text = format!("unexpected error from server {i}: {e}");
                            let err = OrchErr::ServerError(text);
                            let event = TrainingEvent::Error(err);
                            let _ = tx.send(event).await;
                            return;
                        }
                    }
                }

                let model_size = model_params.len();
                info!("received {model_size} total parameters");

                let event = TrainingEvent::Complete(model_params);
                let _ = tx.send(event).await;
            });
        });

        rx
    }

    /// Connects to all parameter servers and sends each its bootstrap spec.
    ///
    /// # Args
    /// * `servers` - List of (address, spec) pairs for each parameter server.
    ///
    /// # Returns
    /// A list of open (receiver, sender) channel pairs, one per server.
    ///
    /// # Errors
    /// Returns an `OrchErr` if any connection or send fails.
    async fn create_servers(servers: Vec<(SocketAddr, ServerSpec)>) -> Result<Vec<(NetRx, NetTx)>> {
        let mut channels = Vec::with_capacity(servers.len());

        for (addr, spec) in servers {
            let (rx, mut tx) = Self::open_channel(addr)
                .await
                .map_err(|source| OrchErr::ConnectionFailed { addr, source })?;

            tx.send(&Msg::Control(Command::CreateServer(spec))).await?;
            channels.push((rx, tx));
        }

        Ok(channels)
    }

    /// Connects to all workers and sends each its bootstrap spec.
    ///
    /// # Args
    /// * `workers` - List of (address, spec) pairs for each worker.
    ///
    /// # Returns
    /// A list of open (receiver, sender) channel pairs, one per worker.
    ///
    /// # Errors
    /// Returns an `OrchestratorError` if any connection or send fails.
    async fn create_workers(workers: Vec<(SocketAddr, WorkerSpec)>) -> Result<Vec<(NetRx, NetTx)>> {
        let mut channels = Vec::with_capacity(workers.len());

        for (addr, spec) in workers {
            log::debug!("connecting to worker at {addr}");

            // TODO: o el path viene hasta acá que no me gusta o se resuelve
            // esto antes y workers deja de ser un vec con address worker
            // spec y es alguna metadata útil para acá mismo...
            let path = Path::new("todo");
            let mut dataset = File::open(path).await.unwrap();
            let dataset_spec = spec.dataset;

            log::info!("worker at {addr} ready");
            let (rx, mut tx) = Self::open_channel(addr)
                .await
                .map_err(|source| OrchErr::ConnectionFailed { addr, source })?;

            tx.send(&Msg::Control(Command::CreateWorker(spec))).await?;

            send_dataset(&mut dataset, dataset_spec, &mut tx);

            channels.push((rx, tx));
        }

        Ok(channels)
    }

    /// Opens a TCP channel to the given address.
    ///
    /// # Args
    /// * `addr` - The socket address to connect to.
    ///
    /// # Returns
    /// A (receiver, sender) channel pair.
    ///
    /// # Errors
    /// Returns an `io::Error` if the connection fails.
    async fn open_channel(addr: SocketAddr) -> io::Result<(NetRx, NetTx)> {
        let stream = TcpStream::connect(addr).await?;
        let (rx, tx) = stream.into_split();
        Ok(comms::channel(rx, tx))
    }
}

fn is_eof(e: &io::Error) -> bool {
    matches!(
        e.kind(),
        io::ErrorKind::UnexpectedEof | io::ErrorKind::ConnectionReset
    ) || e.to_string().contains("early eof")
}
