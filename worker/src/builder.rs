use std::{io, num::NonZeroUsize};

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
    ///
    /// # Panics
    /// Never panics.
    pub fn build(spec: &WorkerSpec) -> io::Result<Worker<Strategy>> {
        let cfg = WorkerConfig::from_spec(spec);
        let strategy = Strategy::from_spec(&spec.strategy)?;
        Ok(Worker::new(
            cfg,
            NonZeroUsize::new(spec.num_params.get()).unwrap(),
            strategy,
        ))
    }
}
