use std::thread;
use std::{net::SocketAddr, path::Path};
use tokio::fs::File;

use comms::{
    OnoReceiver, OnoSender,
    msg::{Command, Msg, Payload},
    send_dataset::send_dataset,
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
    /// Training completed and all servers returned their final model parameters.
    Complete(Vec<f32>),
    /// A worker or server produced an unrecoverable error.
    Error(OrchestratorError),
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
    /// Returns an `OrchestratorError` if any connection or bootstrap message fails.
    pub fn new(
        workers: Vec<(SocketAddr, WorkerSpec)>,
        servers: Vec<(SocketAddr, ServerSpec)>,
    ) -> Result<Self, OrchestratorError> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;

        log::debug!("connecting to servers");
        let server_chans = runtime.block_on(Self::create_servers(servers))?;
        log::debug!("connecting to workers");
        let worker_chans = runtime.block_on(Self::create_workers(workers))?;
        log::info!("all connections established");

        Ok(Self {
            runtime,
            servers: server_chans,
            workers: worker_chans,
        })
    }

    /// Consumes the session and returns a channel receiver of training events.
    ///
    /// Spawns a background thread that drives the session, listening to all
    /// workers and parameter servers, and forwards events through the channel.
    ///
    /// # Returns
    /// A receiver that yields `TrainingEvent`s as training progresses.
    pub fn take_events(self) -> mpsc::Receiver<TrainingEvent> {
        let (tx, rx) = mpsc::channel(256);

        thread::spawn(move || {
            let server_rxs: Vec<NetRx> = self.servers.into_iter().map(|(rx, _)| rx).collect();

            self.runtime.block_on(async move {
                let mut handles = Vec::new();

                for (i, (mut wrx, _wtx)) in self.workers.into_iter().enumerate() {
                    let tx = tx.clone();
                    handles.push(tokio::spawn(async move {
                        let _wtx = _wtx;
                        let mut buf = vec![0u32; 1024];
                        log::debug!("listening to worker {i}");
                        loop {
                            match wrx.recv_into(&mut buf).await {
                                Ok(Msg::Control(Command::ReportLoss { losses })) => {
                                    log::debug!("worker {i} reported {} losses", losses.len());
                                    let _ = tx
                                        .send(TrainingEvent::Loss {
                                            worker_id: i,
                                            losses: losses.to_vec(),
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

                log::debug!("all workers done, reading final params from all servers");
                let mut buf = vec![0u32; 1024];
                let mut all_params: Vec<f32> = Vec::new();

                for (i, mut srx) in server_rxs.into_iter().enumerate() {
                    match srx.recv_into(&mut buf).await {
                        Ok(Msg::Data(Payload::Params(params))) => {
                            log::info!("server {i}: received {} parameters", params.len());
                            all_params.extend_from_slice(params);
                        }
                        Ok(msg) => {
                            log::error!("server {i}: unexpected message {msg:?}");
                            let _ = tx
                                .send(TrainingEvent::Error(OrchestratorError::ServerError(
                                    format!("server {i}: unexpected message {msg:?}"),
                                )))
                                .await;
                            return;
                        }
                        Err(e) => {
                            log::error!("server {i} error: {e}");
                            let _ = tx
                                .send(TrainingEvent::Error(OrchestratorError::ServerError(
                                    format!("server {i}: {e}"),
                                )))
                                .await;
                            return;
                        }
                    }
                }

                log::info!("received {} total parameters", all_params.len());
                let _ = tx.send(TrainingEvent::Complete(all_params)).await;
            });
        });

        rx
    }

    /// Blocks until all workers finish and returns the final model parameters.
    ///
    /// # Returns
    /// The concatenated final parameter vector received from all parameter servers.
    ///
    /// # Errors
    /// Returns an `OrchestratorError` if any worker or server reports a failure.
    pub fn wait(self) -> Result<Vec<f32>, OrchestratorError> {
        self.runtime.block_on(async move {
            let handles: Vec<_> = self
                .workers
                .into_iter()
                .enumerate()
                .map(|(i, (mut rx, _))| {
                    tokio::spawn(async move {
                        let mut buf = vec![0u32; 1024];
                        log::debug!("wait: listening to worker {i}");
                        loop {
                            match rx.recv_into(&mut buf).await {
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

            log::debug!("wait: reading final params from all servers");
            let mut buf = vec![0u32; 1024];
            let mut all_params: Vec<f32> = Vec::new();

            for (i, (mut rx, _)) in self.servers.into_iter().enumerate() {
                match rx.recv_into(&mut buf).await {
                    Ok(Msg::Data(Payload::Params(params))) => {
                        log::info!("wait: server {i} received {} parameters", params.len());
                        all_params.extend_from_slice(params);
                    }
                    Ok(msg) => {
                        return Err(OrchestratorError::ServerError(format!(
                            "server {i}: unexpected message: {msg:?}"
                        )));
                    }
                    Err(e) => {
                        return Err(OrchestratorError::ServerError(format!("server {i}: {e}")));
                    }
                }
            }

            Ok(all_params)
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

            // TODO: o el path viene hasta acá que no me gusta o se resuelve
            // esto antes y workers deja de ser un vec con address worker
            // spec y es alguna metadata útil para acá mismo...
            let dataset = File::open(path).await.unwrap();
            send_dataset(&mut dataset, spec.dataset, &mut tx);

            log::info!("worker at {addr} ready");
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
