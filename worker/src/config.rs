use std::num::NonZeroUsize;

/// Immutable execution bounds for a worker instance.
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    steps: NonZeroUsize,
}

impl WorkerConfig {
    /// Creates a new worker configuration.
    ///
    /// # Args
    /// * `steps` - Number of steps this worker should execute.
    ///
    /// # Returns
    /// A `WorkerConfig` instance.
    pub fn new(steps: NonZeroUsize) -> Self {
        Self { steps }
    }

    /// Returns the total number of execution steps.
    ///
    /// # Returns
    /// The number of steps this worker should execute.
    pub fn steps(&self) -> usize {
        self.steps.get()
    }
}
