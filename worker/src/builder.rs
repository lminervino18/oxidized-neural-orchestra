use comms::specs::worker::WorkerSpec;
use machine_learning::{dataset::DatasetBuilder, training::TrainerBuilder};

use super::worker::Worker;

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
    pub fn build(
        &self,
        spec: WorkerSpec,
        server_sizes: &[usize],
        samples_raw: Vec<f32>,
        labels_raw: Vec<f32>,
    ) -> Worker {
        let dataset_builder = DatasetBuilder::new();
        let dataset = dataset_builder.build_inmem(spec.dataset, samples_raw, labels_raw);
        let trainer_builder = TrainerBuilder::new();
        let trainer = trainer_builder.build(spec.trainer, server_sizes, dataset);
        Worker::new(trainer)
    }
}
