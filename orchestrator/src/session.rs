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
/// Events are delivered through the channel returned by [`Session::events`].
pub struct Session {
    runtime: Runtime,
    events_rx: mpsc::Receiver<TrainingEvent>,
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

        // Channel capacity: workers + server, each can produce multiple events.
        let (tx, events_rx) = mpsc::channel(64);

        for (i, (rx, _tx)) in worker_chans.into_iter().enumerate() {
            let tx = tx.clone();
            runtime.spawn(Self::listen_worker(i, rx, tx));
        }

        // One server for now.
        let (server_rx, _server_tx) = server_chans.remove(0);
        runtime.spawn(Self::listen_server(server_rx, tx));

        Ok(Self { runtime, events_rx })
    }

    /// Returns the receiver side of the training events channel.
    ///
    /// The channel closes once all workers and the server have finished or errored.
    pub fn events(&mut self) -> &mut mpsc::Receiver<TrainingEvent> {
        &mut self.events_rx
    }

    /// Blocks until training completes and returns the final model parameters.
    ///
    /// Convenience method for non-TUI usage. Discards intermediate loss events.
    ///
    /// # Errors
    /// Returns an error if the session ends without receiving final parameters.
    pub fn wait(mut self) -> io::Result<Vec<f32>> {
        self.runtime.block_on(async move {
            while let Some(event) = self.events_rx.recv().await {
                match event {
                    TrainingEvent::Complete(params) => return Ok(params),
                    TrainingEvent::Error(msg) => {
                        return Err(io::Error::other(msg));
                    }
                    _ => {}
                }
            }

            Err(io::Error::other(
                "session ended without receiving final parameters",
            ))
        })
    }

    /// Listens to all messages from a single worker and forwards them as [`TrainingEvent`]s.
    async fn listen_worker(
        worker_id: usize,
        mut rx: NetRx,
        tx: mpsc::Sender<TrainingEvent>,
    ) {
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

    /// Listens to the parameter server and forwards its final response as a [`TrainingEvent`].
    async fn listen_server(mut rx: NetRx, tx: mpsc::Sender<TrainingEvent>) {
        let event = match rx.recv().await {
            Ok(Msg::Data(Payload::Params(params))) => TrainingEvent::Complete(params.to_vec()),
            Ok(msg) => TrainingEvent::Error(format!("server: unexpected message {msg:?}")),
            Err(e) => TrainingEvent::Error(format!("server: {e}")),
        };

        let _ = tx.send(event).await;
    }

    /// Connects to each worker and sends its [`WorkerSpec`].
    async fn connect_workers(
        workers: Vec<(SocketAddr, WorkerSpec)>,
    ) -> io::Result<Vec<(NetRx, NetTx)>> {
        Self::connect_all(workers, |spec| Msg::Control(Command::CreateWorker(spec))).await
    }

    /// Connects to each server and sends its [`ServerSpec`].
    async fn connect_servers(
        servers: Vec<(SocketAddr, ServerSpec)>,
    ) -> io::Result<Vec<(NetRx, NetTx)>> {
        Self::connect_all(servers, |spec| Msg::Control(Command::CreateServer(spec))).await
    }

    /// Connects to a list of addresses, sends the initial message built from each spec, and
    /// returns the open channels.
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

    /// Opens a framed TCP channel to the given address.
    async fn open_channel(addr: SocketAddr) -> io::Result<(NetRx, NetTx)> {
        let stream = TcpStream::connect(addr).await?;
        let (rx, tx) = stream.into_split();
        Ok(comms::channel(rx, tx))
    }
}
