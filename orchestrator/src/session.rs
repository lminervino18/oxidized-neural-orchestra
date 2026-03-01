use std::net::SocketAddr;

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
    runtime: Option<Runtime>,
    events_rx: Option<mpsc::Receiver<TrainingEvent>>,
}

impl Session {
    /// Connects to all workers and servers, sends their specs, and starts listening.
    ///
    /// Order matters:
    /// 1. Connect to server and spawn its listener immediately — so we never
    ///    miss the params message even if training finishes very fast.
    /// 2. Connect to workers and spawn their listeners.
    ///
    /// # Errors
    /// Returns an `io::Error` if any connection or initial handshake fails.
    pub fn new(
        workers: Vec<(SocketAddr, WorkerSpec)>,
        servers: Vec<(SocketAddr, ServerSpec)>,
    ) -> io::Result<Self> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
    .enable_all()
    .build()?;
        let (tx, events_rx) = mpsc::channel(256);

        runtime.block_on(async {
            // Step 1: connect to server and spawn listener FIRST.
            // This ensures the listener is ready before workers trigger training.
            let mut server_chans = Self::connect_servers(servers).await?;
            let (server_rx, _server_tx) = server_chans.remove(0);
            tokio::spawn(Self::listen_server(server_rx, tx.clone()));

            // Step 2: connect to workers and spawn their listeners.
            let worker_chans = Self::connect_workers(workers).await?;
            for (i, (rx, _tx)) in worker_chans.into_iter().enumerate() {
                tokio::spawn(Self::listen_worker(i, rx, tx.clone()));
            }

            Ok::<_, io::Error>(())
        })?;

        Ok(Self {
            runtime: Some(runtime),
            events_rx: Some(events_rx),
        })
    }

    /// Takes the event receiver out of the session for external consumption.
    ///
    /// Returns `None` if already taken.
    pub fn take_events(&mut self) -> Option<mpsc::Receiver<TrainingEvent>> {
        self.events_rx.take()
    }

    /// Blocks until training completes and returns the final model parameters.
    ///
    /// Discards intermediate events. Consumes the session.
    ///
    /// # Errors
    /// Returns an error if the session ends without final parameters,
    /// or if `take_events` was already called before `wait`.
    pub fn wait(mut self) -> io::Result<Vec<f32>> {
        let runtime = self.runtime.take().ok_or_else(|| {
            io::Error::other("runtime already consumed")
        })?;

        let mut rx = self.events_rx.take().ok_or_else(|| {
            io::Error::other("events already taken before wait()")
        })?;

        let result = runtime.block_on(async move {
            while let Some(event) = rx.recv().await {
                match event {
                    TrainingEvent::Complete(params) => return Ok(params),
                    TrainingEvent::Error(msg) => return Err(io::Error::other(msg)),
                    _ => {}
                }
            }
            Err(io::Error::other(
                "session ended without receiving final parameters",
            ))
        });

        runtime.shutdown_timeout(std::time::Duration::from_secs(5));
        result
    }

    async fn listen_worker(worker_id: usize, mut rx: NetRx, tx: mpsc::Sender<TrainingEvent>) {
        loop {
            let event = match rx.recv().await {
                Ok(Msg::Control(Command::ReportLoss { losses, .. })) => {
                    TrainingEvent::Loss { worker_id, losses }
                }
                Ok(Msg::Control(Command::Disconnect)) => {
                    let _ = tx.send(TrainingEvent::WorkerDone(worker_id)).await;
                    return;
                }
                Ok(msg) => TrainingEvent::Error(format!(
                    "worker {worker_id}: unexpected message {msg:?}"
                )),
                Err(e) if Self::is_eof(&e) => {
                    let _ = tx.send(TrainingEvent::WorkerDone(worker_id)).await;
                    return;
                }
                Err(e) => TrainingEvent::Error(format!("worker {worker_id}: {e}")),
            };

            let is_error = matches!(event, TrainingEvent::Error(_));
            let _ = tx.send(event).await;
            if is_error {
                return;
            }
        }
    }

    async fn listen_server(mut rx: NetRx, tx: mpsc::Sender<TrainingEvent>) {
        let event = match rx.recv().await {
            Ok(Msg::Data(Payload::Params(params))) => TrainingEvent::Complete(params.to_vec()),
            Ok(msg) => TrainingEvent::Error(format!("server: unexpected message {msg:?}")),
            Err(e) if Self::is_eof(&e) => {
                TrainingEvent::Error(
                    "server closed connection before sending parameters".into(),
                )
            }
            Err(e) => TrainingEvent::Error(format!("server: {e}")),
        };

        let _ = tx.send(event).await;
    }

    fn is_eof(e: &std::io::Error) -> bool {
        matches!(
            e.kind(),
            std::io::ErrorKind::UnexpectedEof | std::io::ErrorKind::ConnectionReset
        ) || e.to_string().contains("early eof")
    }

    async fn connect_workers(
        workers: Vec<(SocketAddr, WorkerSpec)>,
    ) -> io::Result<Vec<(NetRx, NetTx)>> {
        Self::connect_all(workers, |spec| Msg::Control(Command::CreateWorker(spec))).await
    }

    async fn connect_servers(
        servers: Vec<(SocketAddr, ServerSpec)>,
    ) -> io::Result<Vec<(NetRx, NetTx)>> {
        Self::connect_all(servers, |spec| Msg::Control(Command::CreateServer(spec))).await
    }

    async fn connect_all<S, F>(
        entries: Vec<(SocketAddr, S)>,
        make_msg: F,
    ) -> io::Result<Vec<(NetRx, NetTx)>>
    where
        F: Fn(S) -> Msg<'static>,
    {
        let mut channels = Vec::with_capacity(entries.len());
        for (addr, spec) in entries {
            let (rx, mut tx) = Self::open_channel(addr).await?;
            let msg = make_msg(spec);
            tx.send(&msg).await?;
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
