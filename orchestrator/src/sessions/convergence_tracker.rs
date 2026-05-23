use comms::floats::FloatNonNegative;

use super::GreaterThanOneUsize;

/// Tracks wheather the training is converging or not.
#[derive(Debug)]
pub struct ConvergenceTracker {
    winsize: GreaterThanOneUsize,
    tolerance: FloatNonNegative,
    last: Option<f64>,
    count: usize,
}

impl ConvergenceTracker {
    /// Creates a new `ConvergenceTracker`.
    ///
    /// # Args
    /// * `winsize` - The amount of losses to check.
    /// * `tolerance` - The tolerance of the delta between subsequent losses.
    ///
    /// # Returns
    /// A new `ConvergenceTracker` instance.
    pub fn new(winsize: GreaterThanOneUsize, tolerance: FloatNonNegative) -> Self {
        Self {
            winsize,
            tolerance,
            last: None,
            count: 0,
        }
    }

    /// Records a new loss.
    ///
    /// # Args
    /// * `loss` - The latest loss.
    pub fn record(&mut self, loss: f64) {
        if let Some(last) = self.last {
            let delta = (last - loss).abs();

            if delta > *self.tolerance {
                self.count = 0;
            }
        }

        self.count += 1;
        self.last = Some(loss);
    }

    /// Asks the tracker if the training has converged.
    ///
    /// # Returns
    /// A convergence flag, either `true` if the training converged or `false` otherwise.
    pub fn converged(&self) -> bool {
        *self.winsize + 1 == self.count
    }
}
