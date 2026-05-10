use std::io;

use comms::{
    Acceptor, Connection, Connector, OrchHandle, TransportLayer, get_dataset_cursor,
    specs::{node::NodeSpec, server::ServerSpec, worker::WorkerSpec},
};
use log::{error, info, warn};
use parameter_server::service::ServerBuilder;
use tokio::{
    io::{AsyncRead, AsyncWrite},
    net::tcp::{OwnedReadHalf, OwnedWriteHalf},
};
use worker::{builder::WorkerBuilder, workers::Run};

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
    async fn run_server(&mut self, orch_handle: OrchHandle<T>, spec: ServerSpec) -> io::Result<()> {
        info!("starting parameter server session");

        let mut server_builder = ServerBuilder::new(&mut self.acceptor);
        let mut pserver = server_builder.build(spec, orch_handle).await?;
        pserver.run().await?;

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
            info!("awaiting connection");

            let Ok(conn) = self.acceptor.accept().await else {
                warn!("failed to accept incoming connection");
                continue;
            };

            let Connection::Orchestrator(mut orch_handle) = conn else {
                warn!("expected an orchestrator connection, got something else");
                continue;
            };

            let Ok(node_spec) = orch_handle.pull_specification().await else {
                warn!("failed to get node specification");
                continue;
            };

            if let Err(e) = self.route_spec(orch_handle, node_spec).await {
                error!("session failed with an error: {e}");
            }
        }
    }

    /// Given a node specification calls the correspondent run handler.
    ///
    /// # Args
    /// * `orch_handle` - The handle for communicating with the orchestrator.
    /// * `spec` - The node's specification.
    ///
    /// # Returns
    /// An io error if occurred.
    async fn route_spec(&mut self, orch_handle: OrchHandle<T>, spec: NodeSpec) -> io::Result<()> {
        match spec {
            NodeSpec::Server(spec) => self.run_server(orch_handle, spec).await,
            NodeSpec::Worker(spec) => match self.run_worker(orch_handle, spec).await? {
                Run::Done => Ok(()),
                Run::Switch {
                    server_sizes,
                    server_ordering,
                } => {
                    // 1. Wait for incoming connections from servers
                    // 2. For each new server connection download it's dataset part.
                    // 3. Build the worker, it must be a parameter server worker.
                    // 4. Train.
                    todo!("route_spec match run_worker Run::Switch");
                }
                Run::Upgrade { spec, worker_addrs } => {
                    // 1. Connect to each worker.
                    // 2. For each send it's append it's new dataset partition.
                    // 3. Build the server.
                    // 4. Train.
                    todo!("route_spec match run_worker Run::Upgrade");
                }
            },
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
    ) -> io::Result<Run>
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

        let run = worker.run().await?;
        info!("worker session finished");
        Ok(run)
    }
}
