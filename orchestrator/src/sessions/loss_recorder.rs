use std::{collections::HashSet, num::NonZeroUsize};

/// Records statistics of the worker losses.
#[derive(Debug)]
pub struct LossRecorder {
    losses: HashSet<usize>,
    max_loss: f64,
    sum_loss: f64,
    n: NonZeroUsize,
}

impl LossRecorder {
    /// Creates a new `LossRecorder`.
    ///
    /// # Args
    /// * `n` - The amount of worker losses to track simultaneously.
    ///
    /// # Returns
    /// A new `LossRecorder` instance.
    pub fn new(n: NonZeroUsize) -> Self {
        Self {
            losses: HashSet::with_capacity(n.get()),
            max_loss: 0.0,
            sum_loss: 0.0,
            n,
        }
    }

    /// Records a new loss for a worker.
    ///
    /// # Args
    /// * `worker_id` - The id of the worker whose loss is being recorded.
    /// * `loss` - The latest loss.
    pub fn record(&mut self, worker_id: usize, loss: f64) {
        self.losses.insert(worker_id);
        self.max_loss = self.max_loss.max(loss);
        self.sum_loss += loss;
    }

    /// Gets the max loss of all the workers.
    ///
    /// # Returns
    /// Either `Some(loss)` or `None` if not all workers have recorded losses.
    pub fn max(&self) -> Option<f64> {
        self.stat_if_full(|| self.max_loss)
    }

    /// Gets the mean loss of all the workers.
    ///
    /// # Returns
    /// Either `Some(loss)` or `None` if not all workers have recorded losses.
    pub fn mean(&self) -> Option<f64> {
        self.stat_if_full(|| self.sum_loss / self.n.get() as f64)
    }

    /// Clears the inner state for the following recording.
    pub fn clear(&mut self) {
        self.losses.clear();
        self.max_loss = 0.0;
        self.sum_loss = 0.0;
    }

    /// Gets the statistic if the record is filled with all the workers' losses.
    ///
    /// # Returns
    /// Either `Some(loss)` or `None` if nont all workers have recorded losses.
    fn stat_if_full<F>(&self, f: F) -> Option<f64>
    where
        F: FnOnce() -> f64,
    {
        if self.losses.len() == self.n.get() {
            Some(f())
        } else {
            None
        }
    }
}
