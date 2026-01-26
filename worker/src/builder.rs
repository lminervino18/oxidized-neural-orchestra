use comms::specs::worker::WorkerSpec;

use crate::{Model, Worker, WorkerConfig};

pub struct WorkerBuilder;

impl WorkerBuilder {
    /// Builds a `Worker` from a `WorkerSpec`.
    ///
    /// # Args
    /// * `spec` - Bootstrap specification received from the orchestrator.
    ///
    /// # Returns
    /// A fully initialized `Worker` instance.
    pub fn build(spec: &WorkerSpec) -> Worker<Model> {
        let cfg = WorkerConfig::new(spec.steps);
        let model = Model::from_spec(&spec.model);
        Worker::new(spec.worker_id, cfg, spec.num_params, model)
    }
}
