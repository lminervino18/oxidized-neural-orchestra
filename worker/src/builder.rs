use std::num::NonZeroUsize;

use comms::specs::worker::WorkerSpec;
use ml_core::TrainStrategy;

use crate::{Worker, WorkerConfig};

/// Worker builder.
pub struct WorkerBuilder;

impl WorkerBuilder {
    /// Builds a `Worker` from a `WorkerSpec`.
    ///
    /// # Args
    /// * `spec` - Bootstrap specification received from the orchestrator.
    /// * `strategy` - Concrete `TrainStrategy` implementation for this worker.
    ///
    /// # Returns
    /// A fully initialized `Worker` instance.
    ///
    /// # Errors
    /// Never returns an error.
    ///
    /// # Panics
    /// Never panics.
    pub fn build<S>(spec: &WorkerSpec, strategy: S) -> Worker<S>
    where
        S: TrainStrategy,
    {
        let cfg = WorkerConfig::from_spec(spec);
        Worker::new(cfg, NonZeroUsize::new(spec.num_params.get()).unwrap(), strategy)
    }
}
