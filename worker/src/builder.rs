use comms::specs::worker::WorkerSpec;

use crate::{Strategy, Worker, WorkerConfig};

/// Worker builder.
pub struct WorkerBuilder;

impl WorkerBuilder {
    /// Builds a `Worker` from a `WorkerSpec`.
    ///
    /// # Args
    /// * `spec` - Bootstrap specification received from the orchestrator.
    ///
    /// # Returns
    /// A fully initialized `Worker` instance.
    pub fn build(spec: &WorkerSpec) -> Worker<Strategy> {
        let cfg = WorkerConfig::new(spec.steps);
        let strategy = Strategy::from_spec(&spec.strategy);
        Worker::new(spec.worker_id, cfg, spec.num_params, strategy)
    }
}
