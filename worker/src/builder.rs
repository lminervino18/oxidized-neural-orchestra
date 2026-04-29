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
    cluster_managers::ServerClusterManager,
    workers::{Worker, parameter_server::ParamServerWorker},
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
                server_session_ids,
            } => {
                let mut cluster_manager = ServerClusterManager::new(server_ordering);

                let server_iter = server_addrs
                    .into_iter()
                    .zip(server_sizes.iter().copied())
                    .zip(server_session_ids.iter().copied())
                    .enumerate();

                for (id, ((addr, size), session_id)) in server_iter {
                    let stream = TcpStream::connect(&addr).await?;
                    let (rx, tx) = stream.into_split();
                    let mut server_handle =
                        connector.join_server_session(id, rx, tx, session_id).await?;

                    if let SerializerSpec::SparseCapable { r, seed } = spec.serializer {
                        server_handle.enable_sparse_capability(r, seed);
                    }

                    cluster_manager.spawn(server_handle, size);
                }

                let trainer = trainer_builder.build(spec.trainer, &server_sizes, dataset);
                let worker = ParamServerWorker::new(trainer, cluster_manager, orch_handle);
                Ok(Box::new(worker))
            }
            AlgorithmSpec::AllReduce { .. } => {
                unimplemented!("All Reduce is not yet implemented")
            }
        }
    }
}
