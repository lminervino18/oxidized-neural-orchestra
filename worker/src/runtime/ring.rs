use std::io;

use comms::{
    OnoReceiver, OnoSender,
    msg::{Command, Msg},
    specs::algorithm::RingAllReduceSpec,
};
use log::{info, warn};
use tokio::{
    io::{AsyncRead, AsyncWrite},
    net::{TcpListener, TcpStream},
};

use crate::{
    middleware::ring::RingMiddleware,
    runtime::{DistributedRuntime, OrchRx, OrchTx, RuntimeFuture},
    worker::Worker,
};

/// The ring all-reduce runtime for a worker.
pub struct RingAllReduceRuntime<R, W>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    worker: Worker,
    orch_rx: OnoReceiver<R>,
    orch_tx: OnoSender<W>,
    middleware: RingMiddleware<R, W>,
    ring_addrs: Vec<String>,
}

impl RingAllReduceRuntime<tokio::net::tcp::OwnedReadHalf, tokio::net::tcp::OwnedWriteHalf> {
    /// Bootstraps a TCP-backed `RingAllReduceRuntime`.
    ///
    /// # Args
    /// * `worker` - The local worker state.
    /// * `worker_id` - The id of this worker.
    /// * `spec` - The ring all-reduce algorithm specification.
    /// * `orch_rx` - The receiving end of the communication between the worker and the orchestrator.
    /// * `orch_tx` - The sending end of the communication between the worker and the orchestrator.
    /// * `listener` - The worker listener used to accept ring-neighbor connections.
    ///
    /// # Returns
    /// A bootstrapped `RingAllReduceRuntime`.
    ///
    /// # Errors
    /// Returns an io error if the ring bootstrap fails.
    pub async fn bootstrap(
        worker: Worker,
        worker_id: usize,
        spec: RingAllReduceSpec,
        orch_rx: OrchRx,
        orch_tx: OrchTx,
        listener: TcpListener,
    ) -> io::Result<Self> {
        if spec.worker_addrs.len() < 2 {
            return Err(io::Error::other(
                "ring all-reduce requires at least two workers",
            ));
        }

        let (prev_worker_id, next_worker_id, next_addr) =
            resolve_ring_neighbors(worker_id, &spec.worker_addrs)?;

        let next_stream = TcpStream::connect(next_addr).await?;
        let (next_rx, next_tx) = next_stream.into_split();
        let (next_rx, next_tx) = comms::channel(next_rx, next_tx);

        let (prev_stream, _) = listener.accept().await?;
        let (prev_rx, prev_tx) = prev_stream.into_split();
        let (prev_rx, prev_tx) = comms::channel(prev_rx, prev_tx);

        let ring_addrs = spec.worker_addrs;
        let middleware = RingMiddleware::new(
            prev_worker_id,
            prev_rx,
            prev_tx,
            next_worker_id,
            next_rx,
            next_tx,
        );

        Ok(Self::new(worker, orch_rx, orch_tx, middleware, ring_addrs))
    }
}

impl<R, W> RingAllReduceRuntime<R, W>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    /// Creates a new `RingAllReduceRuntime`.
    ///
    /// # Args
    /// * `worker` - The local worker state.
    /// * `orch_rx` - The receiving end of the communication between the worker and the orchestrator.
    /// * `orch_tx` - The sending end of the communication between the worker and the orchestrator.
    /// * `middleware` - The ring communication middleware.
    ///
    /// # Returns
    /// A new `RingAllReduceRuntime`.
    pub fn new(
        worker: Worker,
        orch_rx: OnoReceiver<R>,
        orch_tx: OnoSender<W>,
        middleware: RingMiddleware<R, W>,
        ring_addrs: Vec<String>,
    ) -> Self {
        Self {
            worker,
            orch_rx,
            orch_tx,
            middleware,
            ring_addrs,
        }
    }

    async fn run_inner(mut self) -> io::Result<()> {
        let _trainer = self.worker.into_trainer();
        let mut rx_buf = vec![0; 1028];

        loop {
            match self.orch_rx.recv_into(&mut rx_buf).await? {
                Msg::Control(Command::Disconnect) => {
                    info!("received a Command::Disconnect from the orchestrator");
                    break;
                }
                other => {
                    warn!("unexpected message from orchestrator, got: {other:?}");
                }
            }
        }

        self.middleware.disconnect().await?;
        let msg = Msg::Control(Command::Disconnect);
        self.orch_tx.send(&msg).await?;
        Err(io::Error::other(
            "ring all-reduce training loop is not implemented yet",
        ))
    }

    /// Runs the runtime to completion.
    ///
    /// # Returns
    /// An io error if the runtime fails.
    pub async fn run(self) -> io::Result<()> {
        self.run_inner().await
    }
}

impl DistributedRuntime
    for RingAllReduceRuntime<tokio::net::tcp::OwnedReadHalf, tokio::net::tcp::OwnedWriteHalf>
{
    fn run(self: Box<Self>) -> RuntimeFuture {
        Box::pin(async move { self.run_inner().await })
    }
}

fn resolve_ring_neighbors<'a>(
    worker_id: usize,
    worker_addrs: &'a [String],
) -> io::Result<(usize, usize, &'a str)> {
    if worker_id >= worker_addrs.len() {
        return Err(io::Error::other(format!(
            "worker {worker_id} is not part of the ring"
        )));
    }

    let ring_len = worker_addrs.len();
    let prev_worker_id = if worker_id == 0 {
        ring_len - 1
    } else {
        worker_id - 1
    };
    let next_worker_id = (worker_id + 1) % ring_len;
    let next_addr = worker_addrs
        .get(next_worker_id)
        .ok_or_else(|| io::Error::other(format!("missing address for worker {next_worker_id}")))?;

    Ok((prev_worker_id, next_worker_id, next_addr))
}
