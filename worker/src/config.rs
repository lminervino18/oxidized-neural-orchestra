use std::num::NonZeroUsize;

/// Immutable worker configuration.
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    pub worker_id: usize,
    pub num_workers: NonZeroUsize,

    /// Total number of model parameters (flat vector length).
    pub num_params: usize,

    /// Training steps (in the current protocol, "epochs" are steps).
    pub steps: usize,

    /// Microbatch accumulation factor (1 = send a gradient every batch/step).
    pub microbatch_k: NonZeroUsize,

    /// Enable dataset prefetch (stage later in `data/dataloader.rs`).
    pub prefetch: bool,
}

impl WorkerConfig {
    pub fn validate(&self) {
        assert!(
            self.worker_id < self.num_workers.get(),
            "worker_id out of range"
        );
        assert!(self.num_params > 0, "num_params must be > 0");
        assert!(self.steps > 0, "steps must be > 0");
    }
}
