use std::collections::HashMap;

/// Tracks wheather the training is converging or not.
pub struct ConvergenceTracker {
    pending: HashMap<usize, f64>,
    n_workers: usize,
    prev_avg: Option<f64>,
}

impl ConvergenceTracker {
    /// Creates a new `ConvergenceTracker`.
    ///
    /// # Args
    /// * `n_workers` - The amount of workers
    ///
    /// # Returns
    /// A new `ConvergenceTracker` instance.
    pub fn new(n_workers: usize) -> Self {
        Self {
            n_workers,
            pending: HashMap::new(),
            prev_avg: None,
        }
    }

    /// Records the loss of the given worker and computes the mean of the loss if
    /// all workers have recorded their last loss.
    ///
    /// # Args
    /// * `worker_id` - The id of the worker whose losses are being recorded.
    /// * `losses` - The losses to record.
    ///
    /// # Returns
    /// An optional (previous, last) loss tuple if all workers have recorded their loss.
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
