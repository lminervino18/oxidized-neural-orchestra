use comms::specs::worker::WorkerSpec;

use super::worker::Worker;
use machine_learning::{dataset::DatasetBuilder, training::TrainerBuilder};

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
    ///
    /// # Returns
    /// A fully initialized `Worker` instance.
    pub fn build(&self, spec: WorkerSpec, server_sizes: &[usize], dataset_raw: Vec<f32>) -> Worker {
        let dataset_builder = DatasetBuilder::new();
        let dataset = dataset_builder.build_inmem(spec.dataset, dataset_raw);
        let trainer_builder = TrainerBuilder::new();
        let trainer = trainer_builder.build(spec.trainer, server_sizes, dataset);
        Worker::new(trainer)
    }
}
