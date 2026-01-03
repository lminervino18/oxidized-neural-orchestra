use std::time::{Duration, Instant};

#[derive(Debug, Default, Clone)]
pub struct WorkerMetrics {
    pub recv_time: Duration,
    pub compute_time: Duration,
    pub send_time: Duration,
    pub steps: u64,
}

impl WorkerMetrics {
    pub fn bump_step(&mut self) {
        self.steps += 1;
    }

    pub fn timed<F, T>(acc: &mut Duration, f: F) -> T
    where
        F: FnOnce() -> T,
    {
        let t0 = Instant::now();
        let out = f();
        *acc += t0.elapsed();
        out
    }
}
