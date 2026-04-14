use comms::specs::worker::WorkerSpec;
use machine_learning::{dataset::DatasetBuilder, training::TrainerBuilder};

use crate::workers::{all_reduce::AllReduceWorker, parameter_server::ParameterServerWorker};

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
