use std::io;

use comms::{
    Acceptor, Connection, Connector, NodeHandle, OrchEvent, OrchHandle, TransportLayer,
    share_dataset,
    specs::{node::NodeSpec, server::ServerSpec, worker::WorkerSpec},
};
use log::{error, info, warn};
use machine_learning::{datasets::Dataset, training::Trainer};
use parameter_server::service::{Server, ServerBuilder};
use tokio::{
    io::{AsyncRead, AsyncWrite},
    net::tcp::{OwnedReadHalf, OwnedWriteHalf},
};
use worker::{
    builder::WorkerBuilder,
    workers::{ParamServerWorker, Run, Worker},
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

            let Connection::Orch(orch_handle) = self.acceptor.accept().await? else {
                warn!("expected an orchestrator connection, got something else");
                continue;
            };

            self.handle_orch(orch_handle).await;
        }
    }

    /// Handles an incoming node connection.
    ///
    /// # Args
    /// * `node_handle` - The newly connected node handle.
    pub async fn handle_node(&mut self, mut node_handle: NodeHandle<T>) {
        todo!()
    }

    /// Handles an incoming orchestrator connection.
    ///
    /// # Args
    /// * `orch_handle` - The newly connected orchestrator handle.
    pub async fn handle_orch(&mut self, mut orch_handle: OrchHandle<T>) {
        let spec = while let Ok(event) = orch_handle.recv_event().await {
            match event {
                OrchEvent::Create { spec } => {
                    if let Err(e) = self.route(spec, orch_handle).await {
                        error!("session failed with an error: {e}");
                    }
                }
                OrchEvent::Disconnect => return,
                OrchEvent::StatsRequest { reqs } => {
                    self.service_stat_requests(reqs).await;
                }
                msg => {
                    warn!("received an unexpected orch event: {msg:?}");
                    continue;
                }
            }
        };
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
        let mut server_builder = ServerBuilder::new(&mut self.acceptor);
        let mut server = server_builder.build(spec, orch_handle).await?;
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
        let mut worker_builder = WorkerBuilder::new(&mut self.acceptor, self.connector.clone());
        let mut worker = worker_builder.build(&spec, &mut orch_handle).await?;

        match self.run_worker(worker.as_mut()).await? {
            Run::Done => Ok(()),
            Run::Switch {
                server_addrs,
                server_sizes,
                server_ordering,
            } => {
                let trainer = worker.into_trainer();
                let mut worker = self
                    .switch(
                        spec,
                        server_addrs,
                        server_sizes,
                        server_ordering,
                        trainer,
                        &mut orch_handle,
                    )
                    .await?;

                worker.run().await.map(|_| ())
            }
            Run::Upgrade { spec } => {
                let trainer = worker.into_trainer();
                let dataset = trainer.into_dataset();
                let mut server = self.upgrade(spec, orch_handle, dataset).await?;
                server.run().await
            }
        }
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

    /// Switches from an `AllReduceWorker` to a `ParamServerWorker`.
    ///
    /// # Args
    /// * `spec` - The worker's previous specification.
    /// * `server_addrs` - The network addresses of the server nodes.
    /// * `server_sizes` - The sizes of each server node.
    /// * `server_ordering` - The ordering of server layers.
    /// * `trainer` - The trainer used to train until now.
    /// * `orch_handle` - The handle for communicating with the orchestrator.
    ///
    /// # Returns
    /// A new worker (presumably a `ParamServerWorker`) or an io error if occurred.
    async fn switch<'a>(
        &mut self,
        spec: WorkerSpec,
        server_addrs: Vec<String>,
        server_sizes: Vec<usize>,
        server_ordering: Vec<usize>,
        trainer: Box<dyn Trainer>,
        orch_handle: &'a mut OrchHandle<T>,
    ) -> io::Result<ParamServerWorker<'a, T>> {
        let mut worker_builder = WorkerBuilder::new(&mut self.acceptor, self.connector.clone());
        let worker = worker_builder
            .build_switched(
                spec,
                server_addrs,
                server_sizes,
                server_ordering,
                trainer,
                orch_handle,
            )
            .await?;

        Ok(worker)
    }

    /// Promotes a worker into a `ParameterServer` by first sharing it's dataset with the new
    /// worker instances.
    ///
    /// # Args
    /// * `spec` - The server specification.
    /// * `orch_handle` - The handle for communicating with the orchestrator.
    /// * `dataset` - The server's previous dataset that's to be shared with the workers.
    ///
    /// # Returns
    /// The server ready to train.
    async fn upgrade(
        &mut self,
        spec: ServerSpec,
        orch_handle: OrchHandle<T>,
        dataset: Dataset,
    ) -> io::Result<Box<dyn Server<T>>> {
        const CHUNK_SIZE: usize = 1 << 12;

        let mut partitions = dataset.partition(spec.nworkers);
        let mut server_builder = ServerBuilder::new(&mut self.acceptor);

        let server = server_builder
            .build_with(spec, orch_handle, async |worker_handle| {
                // SAFETY: The amount of partitions is taken from
                //         the spec, meaning, this closure will be
                //         called at most `spec.nworkers` times.
                let (samples_raw, labels_raw) = partitions.next().unwrap();
                let mut samples_cursor = share_dataset::get_dataset_cursor(samples_raw);
                let mut labels_cursor = share_dataset::get_dataset_cursor(labels_raw);

                worker_handle
                    .push_dataset(
                        &mut samples_cursor,
                        &mut labels_cursor,
                        samples_raw.len(),
                        labels_raw.len(),
                        CHUNK_SIZE,
                    )
                    .await?;

                Ok(())
            })
            .await?;

        Ok(server)
    }
}
