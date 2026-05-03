use std::io;

use comms::{
    Acceptor, Connection, Connector, OrchEvent, OrchHandle, PullSpecResponse, TransportLayer,
    get_dataset_cursor,
    specs::{server::ServerSpec, worker::WorkerSpec},
};
use log::warn;
use parameter_server::service::ServerBuilder;
use tokio::{
    io::{AsyncRead, AsyncWrite},
    net::tcp::{OwnedReadHalf, OwnedWriteHalf},
};
use worker::builder::WorkerBuilder;

/// Routes the first orchestrator connection to the appropriate runtime role.
///
/// A single `NodeRouter` represents one process lifecycle. After receiving a
/// `comms::protocol::NodeSpec` from the orchestrator, the process commits to
/// either a parameter server or worker role for the entire training's duration.
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

        Ok(())
    }
}

impl<T, F, G> NodeRouter<OwnedReadHalf, OwnedWriteHalf, T, F, G>
where
    T: TransportLayer + 'static,
    F: AsyncFn() -> io::Result<T>,
    G: Fn(OwnedReadHalf, OwnedWriteHalf) -> T + Clone,
{
    /// Accepts the orchestrator connection, reads the bootstrap spec, and runs the assigned role.
    ///
    /// # Returns
    /// An io error if occurred.
    pub async fn run(&mut self) -> io::Result<()> {
        let Connection::Orchestrator(mut orch_handle) = self.acceptor.accept().await? else {
            return Err(io::Error::other(
                "expected orchestrator as first connection",
            ));
        };

        match orch_handle.pull_specification().await? {
            PullSpecResponse::ParameterServer(spec) => self.run_server(orch_handle, spec).await,
            PullSpecResponse::Worker(spec) => self.run_worker(orch_handle, spec).await,
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

        worker.run().await
    }
}
