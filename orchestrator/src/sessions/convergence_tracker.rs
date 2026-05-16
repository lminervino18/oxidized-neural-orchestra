use std::collections::HashMap;

pub struct ConvergenceTracker {
    pending: HashMap<usize, f64>,
    n_workers: usize,
    prev_avg: Option<f64>,
}

impl ConvergenceTracker {
    pub fn new(n_workers: usize) -> Self {
        Self {
            n_workers,
            pending: HashMap::new(),
            prev_avg: None,
        }
    }

    pub fn record(&mut self, worker_id: usize, losses: &[f64]) -> Option<(f64, f64)> {
        let last = *losses.last()?;
        self.pending.insert(worker_id, last);

        if self.pending.len() < self.n_workers {
            return None;
        }

        let pending_sum: f64 = self.pending.values().sum();
        let curr = pending_sum / self.n_workers as f64;
        self.pending.clear();

        let signal = self.prev_avg.map(|prev| (prev, curr));
        self.prev_avg = Some(curr);
        signal
    }
}
