use std::net::SocketAddr;
use std::thread;

use comms::{
    OnoReceiver, OnoSender,
    msg::{Command, Msg, Payload},
    specs::{server::ServerSpec, worker::WorkerSpec},
};
use tokio::{
    io,
    net::{
        TcpStream,
        tcp::{OwnedReadHalf, OwnedWriteHalf},
    },
    runtime::Runtime,
    sync::mpsc,
};

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
    Error(String),
}

/// Represents an ongoing training session.
pub struct Session {
    runtime: Runtime,
    server: (NetRx, NetTx),
    workers: Vec<(NetRx, NetTx)>,
}

impl Session {
    pub fn new(
        workers: Vec<(SocketAddr, WorkerSpec)>,
        servers: Vec<(SocketAddr, ServerSpec)>,
    ) -> io::Result<Self> {
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

    /// Takes an event receiver, spawning a background thread that drives
    /// the session and forwards events through the channel.
    ///
    /// Consumes the session — cannot call `wait()` after this.
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
                                    log::debug!(
                                        "worker {i} reported {} losses",
                                        losses.len()
                                    );
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
                                        .send(TrainingEvent::Error(format!(
                                            "worker {i}: unexpected {msg:?}"
                                        )))
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
                                        .send(TrainingEvent::Error(format!(
                                            "worker {i}: {e}"
                                        )))
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
                        TrainingEvent::Error(format!("server: unexpected {msg:?}"))
                    }
                    Err(e) => {
                        log::error!("server error: {e}");
                        TrainingEvent::Error(format!("server: {e}"))
                    }
                };
                let _ = tx.send(event).await;
            });
        });

        rx
    }

    /// Original blocking wait — unchanged from before.
    pub fn wait(self) -> io::Result<Vec<f32>> {
        self.runtime.block_on(async move {
            for (mut rx, _) in self.workers {
                while !matches!(rx.recv().await?, Msg::Control(Command::Disconnect)) {}
            }

            let (mut rx, _) = self.server;

            match rx.recv().await? {
                Msg::Data(Payload::Params(params)) => Ok(params.to_vec()),
                msg => Err(io::Error::other(format!(
                    "received an invalid message: {msg:?}"
                ))),
            }
        })
    }

    async fn create_servers(
        servers: Vec<(SocketAddr, ServerSpec)>,
    ) -> io::Result<Vec<(NetRx, NetTx)>> {
        let mut channels = Vec::with_capacity(servers.len());
        for (addr, spec) in servers {
            log::debug!("connecting to server at {addr}");
            let (rx, mut tx) = Self::open_channel(addr).await?;
            tx.send(&Msg::Control(Command::CreateServer(spec))).await?;
            log::info!("server at {addr} ready");
            channels.push((rx, tx));
        }
        Ok(channels)
    }

    async fn create_workers(
        workers: Vec<(SocketAddr, WorkerSpec)>,
    ) -> io::Result<Vec<(NetRx, NetTx)>> {
        let mut channels = Vec::with_capacity(workers.len());
        for (addr, spec) in workers {
            log::debug!("connecting to worker at {addr}");
            let (rx, mut tx) = Self::open_channel(addr).await?;
            tx.send(&Msg::Control(Command::CreateWorker(spec))).await?;
            log::info!("worker at {addr} ready");
            channels.push((rx, tx));
        }
        Ok(channels)
    }

    async fn open_channel(addr: SocketAddr) -> io::Result<(NetRx, NetTx)> {
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
