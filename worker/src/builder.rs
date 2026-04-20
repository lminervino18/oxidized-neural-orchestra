use std::io;

use comms::{
    OnoReceiver, OnoSender,
    specs::worker::{AlgorithmSpec, SerializerSpec, WorkerSpec},
};
use machine_learning::{dataset::DatasetBuilder, training::TrainerBuilder};
use tokio::net::{
    TcpStream,
    tcp::{OwnedReadHalf, OwnedWriteHalf},
};

use crate::{
    middlewares::{all_reduce::AllReduceMiddleware, parameter_server::ParameterServerMiddleware},
    workers::{all_reduce::AllReduceWorker, parameter_server::ParameterServerWorker},
};

/// A fully built worker ready to be run.
pub enum BuiltWorker {
    ParameterServer(
        ParameterServerWorker,
        ParameterServerMiddleware<OwnedReadHalf, OwnedWriteHalf>,
    ),
    AllReduce(
        AllReduceWorker,
        AllReduceMiddleware<OwnedReadHalf, OwnedWriteHalf>,
    ),
}

impl BuiltWorker {
    /// Runs the built worker with the orchestrator channel.
    ///
    /// # Args
    /// * `rx` - The receiving end of the communication between the worker and the orchestrator.
    /// * `tx` - The sending end of the communication between the worker and the orchestrator.
    ///
    /// # Returns
    /// An io error if occurred.
    pub async fn run(
        self,
        rx: OnoReceiver<OwnedReadHalf>,
        tx: OnoSender<OwnedWriteHalf>,
    ) -> io::Result<()> {
        match self {
            Self::ParameterServer(worker, middleware) => worker.run(rx, tx, middleware).await,
            Self::AllReduce(worker, middleware) => worker.run(rx, tx, middleware).await,
        }
    }
}

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
    pub async fn build(
        &self,
        spec: WorkerSpec,
        local_addr: String,
        samples_raw: Vec<f32>,
        labels_raw: Vec<f32>,
    ) -> io::Result<BuiltWorker> {
        match spec.algorithm.clone() {
            AlgorithmSpec::ParameterServer {
                server_addrs,
                server_sizes,
                server_ordering,
                server_session_ids,
            } => {
                let serializer = spec.serializer.clone();
                let worker =
                    self.build_parameter_server(spec, &server_sizes, samples_raw, labels_raw);
                let mut middleware = ParameterServerMiddleware::new(server_ordering);

                for ((addr, size), session_id) in server_addrs
                    .into_iter()
                    .zip(server_sizes)
                    .zip(server_session_ids)
                {
                    let stream = TcpStream::connect(addr).await?;
                    let (raw_rx, raw_tx) = stream.into_split();

                    let (mut tmp_rx, mut tmp_tx) = comms::channel(raw_rx, raw_tx);
                    tmp_tx
                        .send(&comms::msg::Msg::Control(
                            comms::msg::Command::JoinServer { session_id },
                        ))
                        .await?;
                    let raw_rx = tmp_rx.into_inner();
                    let raw_tx = tmp_tx.into_inner();

                    let (rx, tx) = match serializer {
                        SerializerSpec::Base => comms::channel(raw_rx, raw_tx),
                        SerializerSpec::SparseCapable { r, seed } => {
                            comms::sparse_tx_channel(raw_rx, raw_tx, r, seed)
                        }
                    };

                    middleware.spawn(rx, tx, size);
                }

                Ok(BuiltWorker::ParameterServer(worker, middleware))
            }
            AlgorithmSpec::AllReduce { worker_addrs } => {
                let worker = self.build_all_reduce(
                    spec,
                    local_addr.clone(),
                    worker_addrs.clone(),
                    samples_raw,
                    labels_raw,
                );
                let middleware = AllReduceMiddleware::new(&local_addr, worker_addrs)?;

                Ok(BuiltWorker::AllReduce(worker, middleware))
            }
        }
    }

    /// Builds a `Worker` from a `WorkerSpec`.
    ///
    /// # Args
    /// * `spec` - The specification for a worker.
    /// * `server_sizes` - The amount of parameters for each server.
    /// * `samples_raw` - The dataset samples raw data.
    /// * `labels_raw` - The dataset labels raw data.
    ///
    /// # Returns
    /// A fully initialized `Worker` instance.
    pub fn build_parameter_server(
        &self,
        spec: WorkerSpec,
        server_sizes: &[usize],
        samples_raw: Vec<f32>,
        labels_raw: Vec<f32>,
    ) -> ParameterServerWorker {
        let dataset_builder = DatasetBuilder::new();
        let dataset = dataset_builder.build_inmem(spec.dataset, samples_raw, labels_raw);
        let trainer_builder = TrainerBuilder::new();
        let trainer = trainer_builder.build(spec.trainer, server_sizes, dataset);
        ParameterServerWorker::new(trainer)
    }

    /// Builds an `AllReduceWorker` from a `WorkerSpec`.
    ///
    /// # Args
    /// * `spec` - The specification for a worker.
    /// * `worker_addr` - The advertised address of this worker.
    /// * `worker_addrs` - The addresses of all workers participating in the ring.
    /// * `samples_raw` - The dataset samples raw data.
    /// * `labels_raw` - The dataset labels raw data.
    ///
    /// # Returns
    /// A partially initialized `AllReduceWorker` instance.
    pub fn build_all_reduce(
        &self,
        spec: WorkerSpec,
        worker_addr: String,
        worker_addrs: Vec<String>,
        samples_raw: Vec<f32>,
        labels_raw: Vec<f32>,
    ) -> AllReduceWorker {
        let _ = spec;
        let _ = samples_raw;
        let _ = labels_raw;

        // TODO: Build the local ML state for all-reduce once the interaction between
        //       the worker, the trainer and the ring synchronization is defined.
        AllReduceWorker::new(worker_addr, worker_addrs)
    }
}
