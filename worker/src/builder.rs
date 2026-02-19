use comms::specs::worker::WorkerSpec;

use crate::Worker;
use machine_learning::training::TrainerBuilder;

pub struct WorkerBuilder;

impl WorkerBuilder {
    /// Builds a `Worker` from a `WorkerSpec`.
    ///
    /// # Args
    /// * `spec` - Bootstrap specification received from the orchestrator.
    ///
    /// # Returns
    /// A fully initialized `Worker` instance.
    pub fn build(spec: WorkerSpec) -> Worker {
        let trainer = TrainerBuilder::new().build(spec.trainer);
        Worker::new(spec.worker_id, spec.max_epochs, trainer)
    }
}
