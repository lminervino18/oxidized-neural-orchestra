use std::{collections::HashMap, num::NonZeroUsize};

/// Records statistics of the worker losses.
#[derive(Debug)]
pub struct LossRecorder {
    losses: HashMap<usize, f64>,
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
            losses: HashMap::with_capacity(n.get()),
            n,
        }
    }

    /// Records a new loss for a worker.
    ///
    /// # Args
    /// * `worker_id` - The id of the worker whose loss is being recorded.
    /// * `loss` - The latest loss.
    pub fn record(&mut self, worker_id: usize, loss: f64) {
        if loss.is_normal() || loss == 0.0 {
            self.losses.insert(worker_id, loss);
        }
    }

    /// Gets the max loss of all the workers.
    ///
    /// # Returns
    /// Either `Some(loss)` or `None` if not all workers have recorded losses.
    pub fn max(&self) -> Option<f64> {
        self.check_full()?;
        self.losses.values().copied().max_by(f64::total_cmp)
    }

    /// Gets the mean loss of all the workers.
    ///
    /// # Returns
    /// Either `Some(loss)` or `None` if not all workers have recorded losses.
    pub fn mean(&self) -> Option<f64> {
        self.check_full()?;
        let sum: f64 = self.losses.values().sum();
        Some(sum / self.n.get() as f64)
    }

    /// Clears the inner state for the following recording.
    pub fn clear(&mut self) {
        self.losses.clear();
    }

    /// Checks wheather the inner losses map is full or not.
    ///
    /// # Returns
    /// `Some(())` if it is, `None` otherwise.
    fn check_full(&self) -> Option<()> {
        (self.losses.len() == self.n.get()).then_some(())
    }
}
