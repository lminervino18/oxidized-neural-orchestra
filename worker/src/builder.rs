use std::io;

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
    ///
    /// # Errors
    /// Returns `io::Error` if the requested training strategy cannot be constructed.
    pub fn build(spec: &WorkerSpec) -> io::Result<Worker<Strategy>> {
        let cfg = WorkerConfig::new(spec.steps);
        let strategy = Strategy::from_spec(&spec.strategy)?;
        Ok(Worker::new(spec.worker_id, cfg, strategy))
    }
}
