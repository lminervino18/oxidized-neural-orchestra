use std::num::NonZeroUsize;

use comms::specs::worker::WorkerSpec;

/// Immutable configuration for a worker instance.
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    worker_id: usize,
    steps: NonZeroUsize,
}

impl WorkerConfig {
    /// Creates a new worker configuration.
    ///
    /// # Args
    /// * `worker_id` - Worker identifier assigned by the orchestrator.
    /// * `steps` - Number of steps this worker should execute.
    pub fn new(worker_id: usize, steps: NonZeroUsize) -> Self {
        Self { worker_id, steps }
    }

    /// Builds a `WorkerConfig` from a wire-level `WorkerSpec`.
    ///
    /// # Args
    /// * `spec` - Bootstrap specification received from the orchestrator.
    ///
    /// # Returns
    /// A `WorkerConfig` containing the execution identity and bounds.
    pub fn from_spec(spec: &WorkerSpec) -> Self {
        Self::new(spec.worker_id, spec.steps)
    }

    /// Returns the identifier of this worker.
    pub fn worker_id(&self) -> usize {
        self.worker_id
    }

    /// Returns the total number of execution steps.
    pub fn steps(&self) -> usize {
        self.steps.get()
    }
}
