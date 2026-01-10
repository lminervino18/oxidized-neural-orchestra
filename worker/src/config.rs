use std::num::NonZeroUsize;

/// Immutable configuration for a worker instance.
///
/// This type defines the minimal identity and execution bounds
/// required by the worker runtime, without embedding any model,
/// data, or networking concerns.
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    worker_id: usize,
    steps: NonZeroUsize,
}

impl WorkerConfig {
    /// Creates a new worker configuration.
    ///
    /// `worker_id` is assumed to be valid and assigned by the
    /// surrounding system (e.g. launcher or cluster manager).
    pub fn new(worker_id: usize, steps: NonZeroUsize) -> Self {
        Self { worker_id, steps }
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
