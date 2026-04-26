use std::io;

use comms::{
    Connector, OrchHandle, TransportLayer,
    specs::worker::{AlgorithmSpec, SerializerSpec, WorkerSpec},
};
use machine_learning::{dataset::DatasetBuilder, training::TrainerBuilder};
use tokio::net::{
    TcpStream,
    tcp::{OwnedReadHalf, OwnedWriteHalf},
};

use crate::{
    middlewares::{ServerClusterManager, WorkerRingManager},
    workers::{AllReduceWorker, Worker, parameter_server::ParamServerWorker},
};

#[derive(Default)]
pub struct WorkerBuilder;

impl WorkerBuilder {
    /// Creates a new `WorkerBuilder`.
    ///
    /// # Returns
    /// A new `WorkerBuilder` instance.
    pub fn new() -> Self {
        Self
    }

    /// Builds a `Worker` from a `WorkerSpec`.
    ///
    /// # Args
    /// * `spec` - The specification for a worker.
    /// * `connector` - The network connector.
    /// * `orch_handle` - The handle to communicate with the orchestrator.
    /// * `samples_raw` - The dataset samples raw data.
    /// * `labels_raw` - The dataset labels raw data.
    ///
    /// # Returns
    /// A fully initialized worker ready to start.
    pub async fn build<T, F>(
        &self,
        spec: WorkerSpec,
        connector: Connector<OwnedReadHalf, OwnedWriteHalf, T, F>,
        orch_handle: OrchHandle<T>,
        samples_raw: Vec<f32>,
        labels_raw: Vec<f32>,
    ) -> io::Result<Box<dyn Worker>>
    where
        T: TransportLayer + 'static,
        F: Fn(OwnedReadHalf, OwnedWriteHalf) -> T,
    {
        let dataset_builder = DatasetBuilder::new();
        let dataset = dataset_builder.build_inmem(spec.dataset, samples_raw, labels_raw);
        let trainer_builder = TrainerBuilder::new();

        match spec.algorithm {
            AlgorithmSpec::ParameterServer {
                server_addrs,
                server_sizes,
                server_ordering,
            } => {
                let mut cluster_manager = ServerClusterManager::new(server_ordering);

                for (id, (addr, &size)) in server_addrs.into_iter().zip(&server_sizes).enumerate() {
                    let stream = TcpStream::connect(addr).await?;
                    let (rx, tx) = stream.into_split();
                    let mut server_handle = connector.connect_parameter_server(id, rx, tx).await?;

                    if let SerializerSpec::SparseCapable { r, seed } = spec.serializer {
                        server_handle.enable_sparse_capability(r, seed);
                    }

                    cluster_manager.spawn(server_handle, size);
                }

                let trainer = trainer_builder.build(spec.trainer, &server_sizes, dataset);
                let worker = ParamServerWorker::new(trainer, cluster_manager, orch_handle);
                Ok(Box::new(worker) as Box<dyn Worker>)
            }
            AlgorithmSpec::AllReduce { worker_addrs } => {
                let connect_to_worker = async |id| {
                    let addr = &worker_addrs[id];
                    let stream = TcpStream::connect(addr).await?;
                    let (rx, tx) = stream.into_split();
                    let mut worker_handle = connector.connect_worker(id, rx, tx).await?;

                    if let SerializerSpec::SparseCapable { r, seed } = spec.serializer {
                        worker_handle.enable_sparse_capability(r, seed);
                    }

                    Ok::<_, io::Error>(worker_handle)
                };

                let n = worker_addrs.len();
                let id = spec.worker_id;

                let prev = connect_to_worker((id + n - 1) % n).await?;
                let next = connect_to_worker((id + 1) % n).await?;

                let model_size = 1; // TODO: Somehow get the model size.
                let ring_manager = WorkerRingManager::new(id, worker_addrs, prev, next, model_size);
                let trainer = trainer_builder.build(spec.trainer, &[model_size], dataset);
                let worker = AllReduceWorker::new(trainer, ring_manager, orch_handle);
                Ok(Box::new(worker) as Box<dyn Worker>)
            }
        }
    }
}
