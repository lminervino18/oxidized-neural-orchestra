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
///
/// Spawns concurrent listeners for each worker and the parameter server.
/// Events are delivered through the channel obtained via [`Session::take_events`].
pub struct Session {
    runtime: Runtime,
    events_rx: Option<mpsc::Receiver<TrainingEvent>>,
}

impl Session {
    /// Connects to all workers and servers, sends their specs, and starts listening.
    ///
    /// # Errors
    /// Returns an `io::Error` if any connection or initial handshake fails.
    pub fn new(
        workers: Vec<(SocketAddr, WorkerSpec)>,
        servers: Vec<(SocketAddr, ServerSpec)>,
    ) -> io::Result<Self> {
        let runtime = Runtime::new()?;

        let worker_chans = runtime.block_on(Self::connect_workers(workers))?;
        let mut server_chans = runtime.block_on(Self::connect_servers(servers))?;

        let (tx, events_rx) = mpsc::channel(64);

        for (i, (rx, _tx)) in worker_chans.into_iter().enumerate() {
            let tx = tx.clone();
            runtime.spawn(Self::listen_worker(i, rx, tx));
        }

        let (server_rx, _server_tx) = server_chans.remove(0);
        runtime.spawn(Self::listen_server(server_rx, tx));

        Ok(Self {
            runtime,
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
    /// Discards intermediate loss events. Consumes the session.
    ///
    /// # Errors
    /// Returns an error if the session ends without receiving final parameters.
    pub fn wait(mut self) -> io::Result<Vec<f32>> {
        let mut rx = self
            .events_rx
            .take()
            .expect("events already taken before wait()");

        self.runtime.block_on(async move {
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
        })
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
            Err(e) => TrainingEvent::Error(format!("server: {e}")),
        };

        let _ = tx.send(event).await;
    }

    async fn connect_workers(workers: Vec<(SocketAddr, WorkerSpec)>) -> io::Result<Vec<(NetRx, NetTx)>> {
        Self::connect_all(workers, |spec| Msg::Control(Command::CreateWorker(spec))).await
    }

    async fn connect_servers(servers: Vec<(SocketAddr, ServerSpec)>) -> io::Result<Vec<(NetRx, NetTx)>> {
        Self::connect_all(servers, |spec| Msg::Control(Command::CreateServer(spec))).await
    }

    async fn connect_all<S, F>(entries: Vec<(SocketAddr, S)>, make_msg: F) -> io::Result<Vec<(NetRx, NetTx)>>
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