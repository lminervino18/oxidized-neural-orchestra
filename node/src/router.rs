use std::io;

use comms::{
    Acceptor, Connection, Connector, OrchEvent, OrchHandle, PullSpecResponse, TransportLayer,
    get_dataset_cursor,
    specs::{server::ServerSpec, worker::WorkerSpec},
};
use log::{info, warn};
use parameter_server::service::ServerBuilder;
use tokio::{
    io::{AsyncRead, AsyncWrite},
    net::tcp::{OwnedReadHalf, OwnedWriteHalf},
};
use worker::builder::WorkerBuilder;

/// Routes incoming orchestrator connections to the appropriate runtime role,
/// keeping the process alive across sequential sessions.
///
/// Each session is isolated: state is owned within the session stack frame and
/// dropped naturally when the session ends. The process then loops back to
/// accept the next session.
pub struct NodeRouter<R, W, T, F, G>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
    T: TransportLayer,
    F: AsyncFn() -> io::Result<T>,
    G: Fn(R, W) -> T,
{
    acceptor: Acceptor<T, F>,
    connector: Connector<R, W, T, G>,
}

impl<R, W, T, F, G> NodeRouter<R, W, T, F, G>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
    T: TransportLayer + 'static,
    F: AsyncFn() -> io::Result<T>,
    G: Fn(R, W) -> T + Clone,
{
    /// Creates a new `NodeRouter`.
    ///
    /// # Args
    /// * `acceptor` - The acceptor used to receive incoming connections.
    /// * `connector` - The connector used by workers to reach parameter servers.
    ///
    /// # Returns
    /// A new `NodeRouter` instance.
    pub fn new(acceptor: Acceptor<T, F>, connector: Connector<R, W, T, G>) -> Self {
        Self {
            acceptor,
            connector,
        }
    }

    /// Runs the node as a server instance for a training session.
    ///
    /// # Args
    /// * `orch_handle` - The handle for communicating with the orchestrator.
    /// * `spec` - The specification for building the server.
    ///
    /// # Returns
    /// An io error if occurred.
    async fn run_server(
        &mut self,
        mut orch_handle: OrchHandle<T>,
        spec: ServerSpec,
    ) -> io::Result<()> {
        info!("starting parameter server session");

        let mut server_builder = ServerBuilder::new(&mut self.acceptor);
        let mut pserver = server_builder.build(spec).await?;

        let mut params = pserver.run().await?;
        loop {
            match orch_handle.recv_event().await? {
                OrchEvent::Disconnect => {
                    orch_handle.disconnect().await?;
                    break;
                }
                OrchEvent::RequestParams => orch_handle.push_params(&mut params).await?,
                event => warn!("Unexpected OrchEvent: {event:?}"),
            }
        }

        info!("parameter server session finished");
        Ok(())
    }
}

impl<T, F, G> NodeRouter<OwnedReadHalf, OwnedWriteHalf, T, F, G>
where
    T: TransportLayer + 'static,
    F: AsyncFn() -> io::Result<T>,
    G: Fn(OwnedReadHalf, OwnedWriteHalf) -> T + Clone,
{
    /// Runs the sequential session loop.
    ///
    /// Blocks indefinitely, accepting one orchestrator connection per iteration.
    /// Session-level errors are logged and the loop continues. Returns only if
    /// the listener itself fails with a fatal error.
    ///
    /// # Errors
    /// Returns an io error if the underlying listener fails.
    pub async fn run(mut self) -> io::Result<()> {
        loop {
            info!("waiting for next session");

            let conn = self.acceptor.accept().await?;

            let Connection::Orchestrator(orch_handle) = conn else {
                warn!(
                    "expected orchestrator as first connection, got unexpected connection type; skipping"
                );
                continue;
            };

            info!("accepted orchestrator connection");

            let result = match orch_handle.pull_specification().await {
                Err(e) => Err(e),
                Ok(PullSpecResponse::ParameterServer(spec)) => {
                    self.run_server(orch_handle, spec).await
                }
                Ok(PullSpecResponse::Worker(spec)) => self.run_worker(orch_handle, spec).await,
            };

            match result {
                Ok(()) => info!("session finished, waiting for next session"),
                Err(e) => warn!("session failed: {e}, resetting and waiting for next session"),
            }
        }
    }

    /// Runs the node as a worker instance for a training session.
    ///
    /// # Args
    /// * `orch_handle` - The handle for communicating with the orchestrator.
    /// * `spec` - The specification for building the worker.
    ///
    /// # Returns
    /// An io error if occurred.
    async fn run_worker(
        &mut self,
        mut orch_handle: OrchHandle<T>,
        spec: WorkerSpec,
    ) -> io::Result<()>
    where
        T: TransportLayer + 'static,
    {
        info!("starting worker session");

        let x_size_bytes = spec.dataset.x_size_bytes as usize;
        let y_size_bytes = spec.dataset.y_size_bytes as usize;
        let mut samples_raw = vec![0.0; x_size_bytes / size_of::<f32>()];
        let mut labels_raw = vec![0.0; y_size_bytes / size_of::<f32>()];

        orch_handle
            .pull_dataset(
                &mut get_dataset_cursor(&mut samples_raw),
                &mut get_dataset_cursor(&mut labels_raw),
                x_size_bytes,
                y_size_bytes,
            )
            .await?;

        let mut worker = WorkerBuilder::new(&mut self.acceptor, self.connector.clone())
            .build(spec, orch_handle, samples_raw, labels_raw)
            .await?;

        worker.run().await?;
        info!("worker session finished");
        Ok(())
    }
}
