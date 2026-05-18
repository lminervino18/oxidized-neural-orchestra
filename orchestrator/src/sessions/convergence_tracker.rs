use std::collections::HashMap;

/// Tracks wheather the training is converging or not.
pub struct ConvergenceTracker {
    pending: HashMap<usize, f64>,
    prev: HashMap<usize, f64>,
    n_workers: usize,
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
            prev: HashMap::new(),
        }
    }

    /// Records the last loss for a worker and returns the max per-worker delta
    /// once all workers have reported for the current sync round.
    ///
    /// Returns `None` if not all workers have reported yet, or on the first
    /// complete round (no previous values to compare against).
    pub fn record(&mut self, worker_id: usize, losses: &[f64]) -> Option<f64> {
        let last = *losses.last()?;
        self.pending.insert(worker_id, last);

        if self.pending.len() < self.n_workers {
            return None;
        }

        let max_delta = if self.prev.len() == self.n_workers {
            self.pending
                .iter()
                .filter_map(|(id, &curr)| self.prev.get(id).map(|&prev| (prev - curr).abs()))
                .fold(0.0f64, f64::max)
        } else {
            std::mem::swap(&mut self.prev, &mut self.pending);
            self.pending.clear();
            return None;
        };

        std::mem::swap(&mut self.prev, &mut self.pending);
        self.pending.clear();

        Some(max_delta)
    }
}
