use std::net::SocketAddr;
use std::thread;

use comms::{
    OnoReceiver, OnoSender,
    msg::{Command, Msg, Payload},
    specs::{server::ServerSpec, worker::WorkerSpec},
};
use tokio::{
    net::{
        TcpStream,
        tcp::{OwnedReadHalf, OwnedWriteHalf},
    },
    runtime::Runtime,
    sync::mpsc,
};

use crate::error::OrchestratorError;

type NetRx = OnoReceiver<OwnedReadHalf>;
type NetTx = OnoSender<OwnedWriteHalf>;

/// An event produced during a training session.
#[derive(Debug)]
pub enum TrainingEvent {
    /// A worker completed an epoch and reported its losses.
    Loss { worker_id: usize, losses: Vec<f32> },
    /// A worker finished and disconnected.
    WorkerDone(usize),
    /// Training completed and the server returned the final model parameters.
    Complete(Vec<f32>),
    /// A worker or server produced an unrecoverable error.
    Error(OrchestratorError),
}

/// Represents an ongoing training session.
pub struct Session {
    runtime: Runtime,
    server: (NetRx, NetTx),
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
    /// Returns an `OrchestratorError` if any connection or bootstrap message fails.
    pub fn new(
        workers: Vec<(SocketAddr, WorkerSpec)>,
        servers: Vec<(SocketAddr, ServerSpec)>,
    ) -> Result<Self, OrchestratorError> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;

        log::debug!("connecting to servers");
        let mut server_chans = runtime.block_on(Self::create_servers(servers))?;
        log::debug!("connecting to workers");
        let worker_chans = runtime.block_on(Self::create_workers(workers))?;
        log::info!("all connections established");

        Ok(Self {
            runtime,
            workers: worker_chans,
            server: server_chans.remove(0),
        })
    }

    /// Consumes the session and returns a channel receiver of training events.
    ///
    /// Spawns a background thread that drives the session, listening to all
    /// workers and the parameter server, and forwards events through the channel.
    ///
    /// # Returns
    /// A receiver that yields `TrainingEvent`s as training progresses.
    pub fn take_events(self) -> mpsc::Receiver<TrainingEvent> {
        let (tx, rx) = mpsc::channel(256);

        thread::spawn(move || {
            let (server_rx, _server_tx) = self.server;

            self.runtime.block_on(async move {
                let mut handles = Vec::new();

                for (i, (mut wrx, _wtx)) in self.workers.into_iter().enumerate() {
                    let tx = tx.clone();
                    handles.push(tokio::spawn(async move {
                        let _wtx = _wtx;
                        log::debug!("listening to worker {i}");
                        loop {
                            match wrx.recv().await {
                                Ok(Msg::Control(Command::ReportLoss { losses, .. })) => {
                                    log::debug!("worker {i} reported {} losses", losses.len());
                                    let _ = tx
                                        .send(TrainingEvent::Loss {
                                            worker_id: i,
                                            losses,
                                        })
                                        .await;
                                }
                                Ok(Msg::Control(Command::Disconnect)) => {
                                    log::info!("worker {i} disconnected");
                                    let _ = tx.send(TrainingEvent::WorkerDone(i)).await;
                                    return;
                                }
                                Ok(msg) => {
                                    log::warn!("worker {i}: unexpected message {msg:?}");
                                    let _ = tx
                                        .send(TrainingEvent::Error(
                                            OrchestratorError::WorkerError {
                                                worker_id: i,
                                                msg: format!("unexpected message {msg:?}"),
                                            },
                                        ))
                                        .await;
                                    return;
                                }
                                Err(e) if is_eof(&e) => {
                                    log::info!("worker {i} closed connection");
                                    let _ = tx.send(TrainingEvent::WorkerDone(i)).await;
                                    return;
                                }
                                Err(e) => {
                                    log::error!("worker {i} error: {e}");
                                    let _ = tx
                                        .send(TrainingEvent::Error(
                                            OrchestratorError::WorkerError {
                                                worker_id: i,
                                                msg: e.to_string(),
                                            },
                                        ))
                                        .await;
                                    return;
                                }
                            }
                        }
                    }));
                }

                for h in handles {
                    let _ = h.await;
                }

                log::debug!("all workers done, reading final params from server");
                let mut srx = server_rx;
                let event = match srx.recv().await {
                    Ok(Msg::Data(Payload::Params(params))) => {
                        log::info!("received {} final parameters", params.len());
                        TrainingEvent::Complete(params.to_vec())
                    }
                    Ok(msg) => {
                        log::error!("server: unexpected message {msg:?}");
                        TrainingEvent::Error(OrchestratorError::ServerError(format!(
                            "unexpected message {msg:?}"
                        )))
                    }
                    Err(e) => {
                        log::error!("server error: {e}");
                        TrainingEvent::Error(OrchestratorError::ServerError(e.to_string()))
                    }
                };
                let _ = tx.send(event).await;
            });
        });

        rx
    }

    /// Blocks until all workers finish and returns the final model parameters.
    ///
    /// # Returns
    /// The final parameter vector received from the parameter server.
    ///
    /// # Errors
    /// Returns an `OrchestratorError` if any worker or the server reports a failure.
    pub fn wait(self) -> Result<Vec<f32>, OrchestratorError> {
        self.runtime.block_on(async move {
            let handles: Vec<_> = self
                .workers
                .into_iter()
                .enumerate()
                .map(|(i, (mut rx, _))| {
                    tokio::spawn(async move {
                        log::debug!("wait: listening to worker {i}");
                        loop {
                            match rx.recv().await {
                                Ok(Msg::Control(Command::Disconnect)) => {
                                    log::info!("wait: worker {i} done");
                                    return Ok::<_, OrchestratorError>(());
                                }
                                Ok(_) => {}
                                Err(e) => {
                                    return Err(OrchestratorError::WorkerError {
                                        worker_id: i,
                                        msg: e.to_string(),
                                    });
                                }
                            }
                        }
                    })
                })
                .collect();

            for h in handles {
                h.await
                    .map_err(|e| OrchestratorError::Io(std::io::Error::other(e.to_string())))??;
            }

            let (mut rx, _) = self.server;
            log::debug!("wait: reading final params from server");

            match rx.recv().await {
                Ok(Msg::Data(Payload::Params(params))) => {
                    log::info!("wait: received {} parameters", params.len());
                    Ok(params.to_vec())
                }
                Ok(msg) => Err(OrchestratorError::ServerError(format!(
                    "unexpected message: {msg:?}"
                ))),
                Err(e) => Err(OrchestratorError::ServerError(e.to_string())),
            }
        })
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
    /// Returns an `OrchestratorError` if any connection or send fails.
    async fn create_servers(
        servers: Vec<(SocketAddr, ServerSpec)>,
    ) -> Result<Vec<(NetRx, NetTx)>, OrchestratorError> {
        let mut channels = Vec::with_capacity(servers.len());
        for (addr, spec) in servers {
            log::debug!("connecting to server at {addr}");
            let (rx, mut tx) = Self::open_channel(addr).await.map_err(|e| {
                OrchestratorError::ConnectionFailed {
                    addr: addr.to_string(),
                    source: e,
                }
            })?;
            tx.send(&Msg::Control(Command::CreateServer(spec))).await?;
            log::info!("server at {addr} ready");
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
    async fn create_workers(
        workers: Vec<(SocketAddr, WorkerSpec)>,
    ) -> Result<Vec<(NetRx, NetTx)>, OrchestratorError> {
        let mut channels = Vec::with_capacity(workers.len());
        for (addr, spec) in workers {
            log::debug!("connecting to worker at {addr}");
            let (rx, mut tx) = Self::open_channel(addr).await.map_err(|e| {
                OrchestratorError::ConnectionFailed {
                    addr: addr.to_string(),
                    source: e,
                }
            })?;
            tx.send(&Msg::Control(Command::CreateWorker(spec))).await?;
            log::info!("worker at {addr} ready");
            channels.push((rx, tx));
        }
        Ok(channels)
    }

    /// Opens a raw TCP channel to the given address.
    ///
    /// # Args
    /// * `addr` - The socket address to connect to.
    ///
    /// # Returns
    /// A (receiver, sender) channel pair backed by the TCP stream.
    ///
    /// # Errors
    /// Returns an `io::Error` if the TCP connection fails.
    async fn open_channel(addr: SocketAddr) -> std::io::Result<(NetRx, NetTx)> {
        let stream = TcpStream::connect(addr).await?;
        let (rx, tx) = stream.into_split();
        Ok(comms::channel(rx, tx))
    }
}

fn is_eof(e: &std::io::Error) -> bool {
    matches!(
        e.kind(),
        std::io::ErrorKind::UnexpectedEof | std::io::ErrorKind::ConnectionReset
    ) || e.to_string().contains("early eof")
}
