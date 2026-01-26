use comms::specs::worker::WorkerSpec;

use crate::{optimizer::Optimizer, OptimizerBuilder, Worker, WorkerConfig};

pub struct WorkerBuilder;

impl WorkerBuilder {
    /// Builds a `Worker` from a `WorkerSpec`.
    ///
    /// # Args
    /// * `spec` - Bootstrap specification received from the orchestrator.
    ///
    /// # Returns
    /// A fully initialized `Worker` instance.
    pub fn build(spec: &WorkerSpec) -> Worker<Box<dyn Optimizer>> {
        let cfg = WorkerConfig::new(spec.steps);
        let optimizer = OptimizerBuilder::build(&spec.model, &spec.training);
        Worker::new(
            spec.worker_id,
            cfg,
            spec.num_params,
            optimizer,
            spec.training.offline_steps,
        )
    }
}
