use std::io;

use comms::{
    Connector, OrchHandle, Rtp, TransportLayer,
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

    /// Builds a fully initialized algorithm-specific worker ready to be run.
    ///
    /// # Args
    /// * `spec` - The specification for a worker.
    /// * `local_addr` - The advertised address of this worker.
    /// * `samples_raw` - The dataset samples raw data.
    /// * `labels_raw` - The dataset labels raw data.
    ///
    /// # Returns
    /// A fully initialized worker ready to start.
    pub async fn build<T>(
        &self,
        spec: WorkerSpec,
        connector: Connector,
        samples_raw: Vec<f32>,
        labels_raw: Vec<f32>,
    ) -> io::Result<Box<dyn Worker<T>>>
    where
        T: TransportLayer,
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
                let trainer = trainer_builder.build(spec.trainer, &server_sizes, dataset);
                let mut cluster_manager = ServerClusterManager::new(server_ordering);

                for (id, (addr, size)) in server_addrs.into_iter().zip(server_sizes).enumerate() {
                    let stream = TcpStream::connect(addr).await?;
                    let (rx, tx) = stream.into_split();
                    let mut server_handle = connector.connect_parameter_server(id, rx, tx).await?;

                    if let SerializerSpec::SparseCapable { r, seed } = spec.serializer {
                        server_handle.enable_sparse_capabiliy(r, seed);
                    }

                    cluster_manager.spawn(server_handle, size);
                }

                let worker = ParamServerWorker::new(trainer, cluster_manager);
                Ok(BuiltWorker::ParameterServer(worker))
            }
            AlgorithmSpec::AllReduce { .. } => {
                unimplemented!("All Reduce is not yet implemented")
            }
        }
    }
}
