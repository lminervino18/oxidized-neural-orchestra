use std::num::NonZeroUsize;

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
    ///
    /// # Returns
    /// A `WorkerConfig` instance containing the provided identity and bounds.
    ///
    /// # Panics
    /// Never panics.
    pub fn new(worker_id: usize, steps: NonZeroUsize) -> Self {
        Self { worker_id, steps }
    }

    /// Returns the identifier of this worker.
    ///
    /// # Returns
    /// The worker identifier as assigned by the orchestrator.
    ///
    /// # Panics
    /// Never panics.
    pub fn worker_id(&self) -> usize {
        self.worker_id
    }

    /// Returns the total number of execution steps.
    ///
    /// # Returns
    /// The number of steps this worker should execute.
    ///
    /// # Panics
    /// Never panics.
    pub fn steps(&self) -> usize {
        self.steps.get()
    }
}
