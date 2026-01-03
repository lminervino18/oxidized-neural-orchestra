use std::time::Duration;

#[derive(Debug, Default, Clone)]
pub struct WorkerMetrics {
    pub recv_time: Duration,
    pub compute_time: Duration,
    pub send_time: Duration,

    pub steps: u64,
    pub microbatches: u64,
    pub samples: u64,
}

impl WorkerMetrics {
    #[inline]
    pub fn bump_step(&mut self) {
        self.steps += 1;
    }

    #[inline]
    pub fn add_microbatches(&mut self, n: usize) {
        self.microbatches += n as u64;
    }

    #[inline]
    pub fn add_samples(&mut self, n: usize) {
        self.samples += n as u64;
    }
}
