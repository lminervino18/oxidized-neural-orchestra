use std::num::NonZeroUsize;

use comms::specs::worker::WorkerSpec;

/// Immutable configuration for a worker instance.
///
/// This type defines the minimal identity and execution bounds required
/// by the worker runtime. It intentionally excludes any model, dataset,
/// or networking configuration.
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    worker_id: usize,
    steps: NonZeroUsize,
}

impl WorkerConfig {
    /// Creates a new worker configuration.
    ///
    /// # Args
    /// * `worker_id` - Unique identifier assigned by the surrounding system
    ///   (e.g. launcher or cluster manager).
    /// * `steps` - Number of training steps this worker should execute.
    ///
    /// # Panics
    /// Never panics. Structural invariants are enforced via types.
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
    ///
    /// # Returns
    /// The worker identifier as assigned by the system.
    pub fn worker_id(&self) -> usize {
        self.worker_id
    }

    /// Returns the total number of execution steps.
    ///
    /// # Returns
    /// The number of steps as a `usize`.
    pub fn steps(&self) -> usize {
        self.steps.get()
    }
}
