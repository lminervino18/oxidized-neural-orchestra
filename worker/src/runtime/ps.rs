use std::{borrow::Cow, io};

use comms::{
    OnoReceiver, OnoSender,
    msg::{Command, Msg},
    specs::algorithm::ParameterServerSpec,
};
use log::{debug, info, warn};
use machine_learning::training::TrainResult;
use tokio::{
    io::{AsyncRead, AsyncWrite},
    net::TcpStream,
};

use crate::{
    middleware::ps::ParameterServerMiddleware,
    runtime::{DistributedRuntime, OrchRx, OrchTx, RuntimeFuture},
    worker::Worker,
};

/// The parameter-server runtime for a worker.
pub struct ParameterServerRuntime<R, W>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    worker: Worker,
    orch_rx: OnoReceiver<R>,
    orch_tx: OnoSender<W>,
    middleware: ParameterServerMiddleware<R, W>,
}

impl ParameterServerRuntime<tokio::net::tcp::OwnedReadHalf, tokio::net::tcp::OwnedWriteHalf> {
    /// Bootstraps a TCP-backed `ParameterServerRuntime`.
    ///
    /// # Args
    /// * `worker` - The local worker state.
    /// * `spec` - The parameter-server algorithm specification.
    /// * `orch_rx` - The receiving end of the communication between the worker and the orchestrator.
    /// * `orch_tx` - The sending end of the communication between the worker and the orchestrator.
    ///
    /// # Returns
    /// A bootstrapped `ParameterServerRuntime`.
    ///
    /// # Errors
    /// Returns an io error if any parameter server connection fails.
    pub async fn bootstrap(
        worker: Worker,
        spec: ParameterServerSpec,
        orch_rx: OrchRx,
        orch_tx: OrchTx,
    ) -> io::Result<Self> {
        let mut middleware = ParameterServerMiddleware::new(spec.server_ordering);

        for (addr, size) in spec.server_addrs.into_iter().zip(spec.server_sizes) {
            let stream = TcpStream::connect(addr).await?;
            let (rx, tx) = stream.into_split();
            let (rx, tx) = comms::channel(rx, tx);
            middleware.spawn(rx, tx, size);
        }

        Ok(Self::new(worker, orch_rx, orch_tx, middleware))
    }
}

impl<R, W> ParameterServerRuntime<R, W>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    /// Creates a new `ParameterServerRuntime`.
    ///
    /// # Args
    /// * `worker` - The local worker state.
    /// * `orch_rx` - The receiving end of the communication between the worker and the orchestrator.
    /// * `orch_tx` - The sending end of the communication between the worker and the orchestrator.
    /// * `middleware` - The parameter-server communication middleware.
    ///
    /// # Returns
    /// A new `ParameterServerRuntime`.
    pub fn new(
        worker: Worker,
        orch_rx: OnoReceiver<R>,
        orch_tx: OnoSender<W>,
        middleware: ParameterServerMiddleware<R, W>,
    ) -> Self {
        Self {
            worker,
            orch_rx,
            orch_tx,
            middleware,
        }
    }

    async fn run_inner(mut self) -> io::Result<()> {
        let mut trainer = self.worker.into_trainer();
        let mut rx_buf = vec![0; 1028];
        let mut should_continue = true;

        while should_continue {
            tokio::select! {
                ret = self.middleware.pull_params() => {
                    debug!("received parameters from all servers, training...");

                    let mut param_manager = ret?;
                    let TrainResult { losses, was_last } = trainer
                        .train(&mut param_manager)
                        .map_err(io::Error::other)?;

                    self.middleware.push_grads().await?;

                    should_continue = !was_last;
                    let msg = Msg::Control(Command::ReportLoss {
                        losses: Cow::Borrowed(losses),
                    });
                    self.orch_tx.send(&msg).await?;
                }
                ret = self.orch_rx.recv_into(&mut rx_buf) => match ret? {
                    Msg::Control(Command::Disconnect) => {
                        info!("received a Command::Disconnect from the orchestrator");
                        break;
                    }
                    other => {
                        warn!("unexpected message from orchestrator, got: {other:?}");
                    }
                }
            }
        }

        self.middleware.disconnect().await?;
        let msg = Msg::Control(Command::Disconnect);
        self.orch_tx.send(&msg).await?;
        Ok(())
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
    for ParameterServerRuntime<tokio::net::tcp::OwnedReadHalf, tokio::net::tcp::OwnedWriteHalf>
{
    fn run(self: Box<Self>) -> RuntimeFuture {
        Box::pin(async move { self.run_inner().await })
    }
}
