use std::{collections::HashMap, num::NonZeroUsize};

/// Records statistics of the worker losses.
#[derive(Debug, Default)]
pub struct LossRecorder {
    losses: HashMap<usize, f64>,
}

impl LossRecorder {
    /// Creates a new `LossRecorder`.
    ///
    /// # Returns
    /// A new `LossRecorder` instance.
    pub fn new() -> Self {
        Self::default()
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
    /// # Args
    /// * `workers` - The amount of workers currently expected to report.
    ///
    /// # Returns
    /// Either `Some(loss)` or `None` if not all workers have recorded losses.
    pub fn max(&self, workers: NonZeroUsize) -> Option<f64> {
        self.check_full(workers)?;
        self.losses.values().copied().max_by(f64::total_cmp)
    }

    /// Gets the mean loss of all the workers.
    ///
    /// # Args
    /// * `workers` - The amount of workers currently expected to report.
    ///
    /// # Returns
    /// Either `Some(loss)` or `None` if not all workers have recorded losses.
    pub fn mean(&self, workers: NonZeroUsize) -> Option<f64> {
        self.check_full(workers)?;
        let sum: f64 = self.losses.values().sum();
        Some(sum / workers.get() as f64)
    }

    /// Clears the inner state for the following recording.
    pub fn clear(&mut self) {
        self.losses.clear();
    }

    /// Checks wheather the inner losses map is full or not.
    ///
    /// # Args
    /// * `workers` - The amount of workers currently expected to report.
    ///
    /// # Returns
    /// `Some(())` if it is, `None` otherwise.
    fn check_full(&self, workers: NonZeroUsize) -> Option<()> {
        (self.losses.len() == workers.get()).then_some(())
    }
}
