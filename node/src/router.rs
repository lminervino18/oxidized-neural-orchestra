use std::io;

use comms::{
    Acceptor, Connection, Connector, OrchHandle, TransportLayer,
    specs::{node::NodeSpec, server::ServerSpec, worker::WorkerSpec},
};
use log::{error, info, warn};
use machine_learning::training::Trainer;
use parameter_server::service::{Server, ServerBuilder};
use tokio::{
    io::{AsyncRead, AsyncWrite},
    net::tcp::{OwnedReadHalf, OwnedWriteHalf},
};
use worker::{
    builder::WorkerBuilder,
    workers::{Run, Worker},
};

/// Routes incoming orchestrator connections to the appropriate runtime role
/// keeping the process alive across sequential sessions.
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
}

impl<T, F, G> NodeRouter<OwnedReadHalf, OwnedWriteHalf, T, F, G>
where
    T: TransportLayer + 'static,
    F: AsyncFn() -> io::Result<T>,
    G: Fn(OwnedReadHalf, OwnedWriteHalf) -> T + Clone,
{
    /// Waits for incoming connections and runs the specified node instance.
    ///
    /// Blocks indefinitely, accepting one orchestrator connection per iteration.
    /// Session-level errors are logged and the loop continues.
    ///
    /// # Returns
    /// An io error if there's an issue accepting new incoming connections.
    pub async fn run(mut self) -> io::Result<()> {
        loop {
            info!("awaiting connection");

            let Connection::Orchestrator(mut orch_handle) = self.acceptor.accept().await? else {
                warn!("expected an orchestrator connection, got something else");
                continue;
            };

            let Ok(spec) = orch_handle.pull_specification().await else {
                warn!("failed to get node specification");
                continue;
            };

            if let Err(e) = self.route(spec, orch_handle).await {
                error!("session failed with an error: {e}");
            }
        }
    }

    /// Given a node specification calls the correspondent run handler.
    ///
    /// # Args
    /// * `spec` - The node's specification.
    /// * `orch_handle` - The handle for communicating with the orchestrator.
    ///
    /// # Returns
    /// An io error if occurred.
    async fn route(&mut self, spec: NodeSpec, orch_handle: OrchHandle<T>) -> io::Result<()> {
        match spec {
            NodeSpec::Server(spec) => self.as_server(spec, orch_handle).await,
            NodeSpec::Worker(spec) => self.as_worker(spec, orch_handle).await,
        }
    }

    /// Builds and runs the node as a server instance.
    ///
    /// # Args
    /// * `spec` - The server's specification.
    /// * `orch_handle` - The handle for communicating with the orchestrator.
    ///
    /// # Returns
    /// An io error if occurred.
    async fn as_server(&mut self, spec: ServerSpec, orch_handle: OrchHandle<T>) -> io::Result<()> {
        let mut server = self.build_server(spec, orch_handle).await?;
        self.run_server(server.as_mut()).await
    }

    /// Builds and runs the node as a worker instance.
    ///
    /// # Args
    /// * `spec` - The worker's specification.
    /// * `orch_handle` - The handle for communicating with the orchestrator.
    ///
    /// # Returns
    /// An io error if occurred.
    async fn as_worker(
        &mut self,
        spec: WorkerSpec,
        mut orch_handle: OrchHandle<T>,
    ) -> io::Result<()> {
        let mut worker = self.build_worker(spec, &mut orch_handle).await?;

        match self.run_worker(worker.as_mut()).await? {
            Run::Done => Ok(()),
            Run::Switch {
                server_addrs,
                server_sizes,
                server_ordering,
            } => {
                let trainer = worker.into_trainer();
                self.switch(
                    server_addrs,
                    server_sizes,
                    server_ordering,
                    trainer,
                    orch_handle,
                )
                .await
            }
            Run::Upgrade { spec } => {
                let trainer = worker.into_trainer();
                self.upgrade(spec, orch_handle, trainer).await
            }
        }
    }

    /// Builds a server given a specification and an orchestrator handle.
    ///
    /// # Args
    /// * `spec` - The specification for building the server.
    /// * `orch_handle` - The handle for communicating with the orchestrator.
    ///
    /// # Returns
    /// An io error if occurred.
    async fn build_server(
        &mut self,
        spec: ServerSpec,
        orch_handle: OrchHandle<T>,
    ) -> io::Result<Box<dyn Server<T>>> {
        let mut server_builder = ServerBuilder::new(&mut self.acceptor);
        server_builder.build(spec, orch_handle).await
    }

    /// Builds a worker given a specification and an orchestrator handle.
    ///
    /// # Args
    /// * `spec` - The specification for building the worker.
    /// * `orch_handle` - The handle for communicating with the orchestrator.
    ///
    /// # Returns
    /// An io error if occurred.
    async fn build_worker<'a>(
        &mut self,
        spec: WorkerSpec,
        orch_handle: &'a mut OrchHandle<T>,
    ) -> io::Result<Box<dyn Worker + 'a>> {
        const UNIT_SIZE: usize = size_of::<f32>();
        let x_size_bytes = spec.dataset.x_size_bytes as usize;
        let y_size_bytes = spec.dataset.y_size_bytes as usize;
        let mut samples_raw = vec![0.0; x_size_bytes / UNIT_SIZE];
        let mut labels_raw = vec![0.0; y_size_bytes / UNIT_SIZE];

        orch_handle
            .pull_dataset(
                &mut comms::get_dataset_cursor(&mut samples_raw),
                &mut comms::get_dataset_cursor(&mut labels_raw),
                x_size_bytes,
                y_size_bytes,
            )
            .await?;

        let worker = WorkerBuilder::new(&mut self.acceptor, self.connector.clone())
            .build(spec, orch_handle, samples_raw, labels_raw)
            .await?;

        Ok(worker)
    }

    /// Runs the node as a server instance for a training session.
    ///
    /// # Args
    /// * `server` - The server to run.
    ///
    /// # Returns
    /// An io error if occurred.
    async fn run_server(&mut self, server: &mut dyn Server<T>) -> io::Result<()> {
        info!("starting parameter server session");
        server.run().await?;
        info!("parameter server session finished");
        Ok(())
    }

    /// Runs the node as a worker instance for a training session.
    ///
    /// # Args
    /// * `orch_handle` - The handle for communicating with the orchestrator.
    /// * `spec` - The specification for building the worker.
    ///
    /// # Returns
    /// An io error if occurred.
    async fn run_worker(&mut self, worker: &mut dyn Worker) -> io::Result<Run>
    where
        T: TransportLayer + 'static,
    {
        info!("starting worker session");
        let run = worker.run().await?;
        info!("worker session finished");
        Ok(run)
    }

    async fn switch(
        &mut self,
        server_addrs: Vec<String>,
        server_sizes: Vec<usize>,
        server_ordering: Vec<usize>,
        trainer: Box<dyn Trainer>,
        orch_handle: OrchHandle<T>,
    ) -> io::Result<()> {
        // 1. Build the worker from the trainer (add WorkerBuilder::with_trainer).
        // 2. Connect to the servers.
        // 3. For each new server connection download it's dataset part.
        // 4. Build the worker, it must be a parameter server worker.
        // 5. Train.
        //
        // TODO: Ver como se puede recibir el dataset de los servers.
        todo!("route_spec match run_worker Run::Switch");
    }

    async fn upgrade(
        &mut self,
        spec: ServerSpec,
        orch_handle: OrchHandle<T>,
        trainer: Box<dyn Trainer>,
    ) -> io::Result<()> {
        // 1. Build the server.
        // 2. For each send and append it's new dataset partition.
        // 3. Train.
        //
        // TODO: Ver como hacer para que el server mande datasets a los workers.
        //       Habría que ver que se puede hacer desde `Self::switch` para esto.
        //
        //       Seguro va a haber que agregar un append dataset al trainer.
        todo!("route_spec match run_worker Run::Upgrade");
    }
}
