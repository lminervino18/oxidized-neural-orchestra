use std::io;

use comms::{
    Acceptor, Connection, Connector, DatasetSrc, OrchHandle, ParamServerHandle, TransportLayer,
    protocol::Entity,
    specs::{
        machine_learning::TrainerSpec,
        worker::{AlgorithmSpec, SerializerSpec, WorkerSpec},
    },
};
use machine_learning::{
    datasets::{DataSrc, Dataset},
    initialization::ParamGenBuilder,
    training::TrainerBuilder,
};
use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};

use crate::{
    middlewares::{ServerClusterManager, WorkerRingManager},
    workers::{AllReduceWorker, Worker, parameter_server::ParamServerWorker},
};

type R = OwnedReadHalf;
type W = OwnedWriteHalf;

/// The worker builder, given a spec, will build a new worker ready to use.
pub struct WorkerBuilder<'a, T, F, G, Fut>
where
    T: TransportLayer,
    F: Fn(R, W) -> T,
    G: Fn() -> Fut,
    Fut: Future<Output = io::Result<(R, W)>>,
{
    acceptor: &'a mut Acceptor<T, F, G, Fut>,
    connector: Connector<T, F>,
}

impl<'a, T, F, G, Fut> WorkerBuilder<'a, T, F, G, Fut>
where
    T: TransportLayer,
    F: Fn(R, W) -> T,
    G: Fn() -> Fut,
    Fut: Future<Output = io::Result<(R, W)>>,
{
    /// Creates a new `WorkerBuilder`.
    ///
    /// # Args
    /// * `acceptor` - The network acceptor.
    /// * `connector` - The network connector.
    ///
    /// # Returns
    /// A new `WorkerBuilder` instance.
    pub fn new(acceptor: &'a mut Acceptor<T, F, G, Fut>, connector: Connector<T, F>) -> Self {
        Self {
            acceptor,
            connector,
        }
    }
}

impl<T, F, G, Fut> WorkerBuilder<'_, T, F, G, Fut>
where
    T: TransportLayer + 'static,
    F: Fn(R, W) -> T,
    G: Fn() -> Fut,
    Fut: Future<Output = io::Result<(R, W)>>,
{
    /// Builds a `Worker` from a `WorkerSpec`.
    ///
    /// # Args
    /// * `spec` - The specification for a worker.
    /// * `orch_handle` - The handle to communicate with the orchestrator.
    ///
    /// # Returns
    /// A fully initialized worker ready to train.
    pub async fn build<'a>(
        &mut self,
        spec: &WorkerSpec,
        orch_handle: &'a mut OrchHandle<T>,
    ) -> io::Result<Box<dyn Worker + 'a>> {
        let data_src = self.download_dataset(orch_handle).await?;
        let trainer_builder = TrainerBuilder::new();

        let WorkerSpec {
            ref trainer,
            ref algorithm,
            serializer,
            seed,
        } = *spec;

        match *algorithm {
            AlgorithmSpec::ParameterServer {
                ref server_addrs,
                ref server_sizes,
                ref server_ordering,
            } => {
                let cluster_manager = self
                    .connect_to_servers(
                        server_addrs,
                        server_sizes,
                        server_ordering.clone(),
                        serializer,
                        seed,
                        async |_| Ok(()),
                    )
                    .await?;

                let mut trainer = trainer_builder.build(trainer.clone(), server_sizes);
                trainer.load_dataset(data_src);

                let worker = ParamServerWorker::new(trainer, cluster_manager, orch_handle);
                Ok(Box::new(worker) as Box<dyn Worker>)
            }
            AlgorithmSpec::AllReduce {
                pos,
                ref worker_addrs,
                ref param_gen,
                amount_of_layers,
            } => {
                let param_gen_builder = ParamGenBuilder::new();
                let mut param_gen = param_gen_builder
                    .build(param_gen.clone(), spec.seed)
                    .map_err(io::Error::other)?;

                let model_size = param_gen.size();
                let ring_manager = self
                    .connect_to_workers(
                        pos,
                        worker_addrs.clone(),
                        model_size,
                        serializer,
                        seed,
                        amount_of_layers,
                    )
                    .await?;

                // SAFETY: The parameter generator was just created.
                let params = param_gen.sample_remaining().unwrap();
                let mut trainer = trainer_builder.build(trainer.clone(), &[model_size]);
                trainer.load_dataset(data_src);

                let worker = AllReduceWorker::new(trainer, ring_manager, orch_handle, params);
                Ok(Box::new(worker) as Box<dyn Worker>)
            }
        }
    }

    /// Builds a `ParameterServerWorker` from an `AllReduceWorker`'s left over data.
    ///
    /// # Args
    /// * `spec` - The worker's old specification.
    /// * `server_addrs` - The servers' network addresses.
    /// * `server_sizes` - The sizes of the worker nodes.
    /// * `server_ordering` - The ordering of the server layers.
    /// * `dataset` - The dataset that the worker has been using until now.
    /// * `trainer_spec` - The trainer specification for the new trainer.
    /// * `orch_handle` - The handle for communicating with the orchestrator.
    ///
    /// # Returns
    /// A new `ParamServerWorker` or an io error if occurred.
    pub async fn build_switched<'a>(
        &mut self,
        spec: WorkerSpec,
        server_addrs: Vec<String>,
        server_sizes: Vec<usize>,
        server_ordering: Vec<usize>,
        dataset: Dataset,
        trainer_spec: TrainerSpec,
        orch_handle: &'a mut OrchHandle<T>,
    ) -> io::Result<ParamServerWorker<'a, T>> {
        let WorkerSpec {
            serializer, seed, ..
        } = spec;

        let trainer_builder = TrainerBuilder::new();
        let mut trainer = trainer_builder.build(trainer_spec, &server_sizes);
        trainer.load_dataset(dataset.into_src());

        let cluster_manager = self
            .connect_to_servers(
                &server_addrs,
                &server_sizes,
                server_ordering,
                serializer,
                seed,
                async |param_handle| {
                    let data_src = self.download_dataset(param_handle).await?;
                    trainer.load_dataset(data_src);
                    Ok(())
                },
            )
            .await?;

        let worker = ParamServerWorker::new(trainer, cluster_manager, orch_handle);
        Ok(worker)
    }

    /// Downloads the dataset from the given dataset source.
    ///
    /// # Args
    /// * `dataset_src` - The handle to communicate with the dataset producer.
    ///
    /// # Returns
    /// A new `Dataset` instance or an io error if occurred.
    async fn download_dataset<S>(&self, dataset_src: &mut S) -> io::Result<DataSrc>
    where
        S: DatasetSrc,
    {
        let (mut xs, mut ys) = (Vec::new(), Vec::new());
        dataset_src.pull_dataset(&mut xs, &mut ys).await?;
        Ok(DataSrc::inmem(xs, ys))
    }

    /// Connects this worker to all the servers in the network.
    ///
    /// # Args
    /// * `server_addrs` - The network addresses of the servers.
    /// * `server_sizes` - The sizes of the servers in amount of parameters they hold.
    /// * `server_ordering` - The ordering of the servers for the layers of the model.
    /// * `serializer_spec` - The spec of the serialization protocol.
    /// * `seed` - An optional seed for the serializer's random number generator.
    ///
    /// # Returns
    /// A new `ServerClusterManager` instance or an io error if occurred.
    async fn connect_to_servers<H>(
        &self,
        server_addrs: &[String],
        server_sizes: &[usize],
        server_ordering: Vec<usize>,
        serializer_spec: SerializerSpec,
        seed: Option<u64>,
        mut connection_hook: H,
    ) -> io::Result<ServerClusterManager<T>>
    where
        H: AsyncFnMut(&mut ParamServerHandle<T>) -> io::Result<()>,
    {
        let mut cluster_manager = ServerClusterManager::new(server_ordering);
        let src = Entity::Worker;

        for (addr, &size) in server_addrs.iter().cloned().zip(server_sizes) {
            let mut server_handle = self.connector.connect_parameter_server(&addr, src).await?;

            if let SerializerSpec::SparseCapable { r } = serializer_spec {
                server_handle.enable_sparse_capability(r, seed);
            }

            connection_hook(&mut server_handle).await?;
            cluster_manager.spawn(server_handle, size);
        }

        Ok(cluster_manager)
    }

    /// Connects this worker to it's previous and next workers in the network.
    ///
    /// # Args
    /// * `pos` - The position of this worker with respect to every other in the ring.
    /// * `worker_addrs` - The addresses of all the workers in the network.
    /// * `model_size` - The amount of parameters of the model.
    /// * `serializer_spec` - The spec of the serialization protocol.
    /// * `seed` - An optional seed for the serializer's random number generator.
    /// * `amount_of_layers` - The amount of layers in the model.
    ///
    /// # Returns
    /// A new `WorkerRingManager` instance or an error if occurred.
    async fn connect_to_workers(
        &mut self,
        pos: usize,
        addrs: Vec<String>,
        model_size: usize,
        serializer_spec: SerializerSpec,
        seed: Option<u64>,
        amount_of_layers: usize,
    ) -> io::Result<WorkerRingManager<T>> {
        let src = Entity::Worker;

        let prev_conn_fut = async {
            loop {
                if let Connection::Worker(worker_handle) = self.acceptor.accept(src).await? {
                    return Ok::<_, io::Error>(worker_handle);
                }
            }
        };

        let n = addrs.len();

        let next_conn_fut = async {
            let addr = &addrs[(pos + 1) % n];
            let mut worker_handle = self.connector.connect_worker(addr, src).await?;

            if let SerializerSpec::SparseCapable { r } = serializer_spec {
                worker_handle.enable_sparse_capability(r, seed);
            }

            Ok::<_, io::Error>(worker_handle)
        };

        let (prev, next) = tokio::try_join!(prev_conn_fut, next_conn_fut)?;
        let ring_manager =
            WorkerRingManager::new(pos, addrs, prev, next, model_size, amount_of_layers);

        Ok(ring_manager)
    }
}
