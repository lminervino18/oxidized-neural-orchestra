use comms::specs::worker::WorkerSpec;

use super::worker::Worker;
use machine_learning::training::TrainerBuilder;

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
    /// # Arguments
    /// * `spec` - The specification for a worker.
    ///
    /// # Returns
    /// A fully initialized `Worker` instance.
    pub fn build(&self, spec: WorkerSpec, server_sizes: &[usize]) -> Worker {
        let trainer_builder = TrainerBuilder::new();
        let trainer = trainer_builder.build(spec.trainer, server_sizes);
        Worker::new(trainer)
    }
}
