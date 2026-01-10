use std::num::NonZeroUsize;

/// Infrastructure-level worker configuration.
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    worker_id: usize,
    steps: NonZeroUsize,
}

impl WorkerConfig {
    pub fn new(worker_id: usize, steps: NonZeroUsize) -> Self {
        Self { worker_id, steps }
    }

    pub fn worker_id(&self) -> usize {
        self.worker_id
    }

    pub fn steps(&self) -> usize {
        self.steps.get()
    }
}
