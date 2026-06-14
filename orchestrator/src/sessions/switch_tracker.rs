use std::collections::VecDeque;

use super::GreaterThanOneUsize;

/// The entity responsible of detecting and engaging the algorithm switch
/// when training under Strategy Switch.
#[derive(Debug)]
pub struct SwitchTracker {
    winsize: GreaterThanOneUsize,
    threshold: f64,
    losses: VecDeque<f64>,
}

impl SwitchTracker {
    /// Creates a new `SwitchTracker`.
    ///
    /// # Args
    /// * `winsize` - The amount of losses to acumulate (paper uses `6`).
    /// * `threshold` - The trigger value for `s` (paper uses `0.01`).
    ///
    /// # Returns
    /// A new `SwitchTracker` instance.
    pub fn new(winsize: GreaterThanOneUsize, threshold: f64) -> Self {
        Self {
            winsize,
            threshold,
            losses: VecDeque::with_capacity(*winsize),
        }
    }

    /// Records a new loss onto the tracker and decides whether to
    /// switch algorithms or continue training under the same algorithm.
    ///
    /// # Args
    /// * `loss` - The new loss to accumulate.
    pub fn record(&mut self, loss: f64) {
        if self.losses.len() == *self.winsize {
            self.losses.pop_front();
        }

        self.losses.push_back(loss);
    }

    /// Determines when the switch should occur.
    ///
    /// # Returns
    /// `true` if it should switch, `false` otherwise.
    pub fn should_switch(&self) -> bool {
        if self.losses.len() < *self.winsize {
            return false;
        }

        let mut sum = 0.0;

        for i in 1..*self.winsize {
            let delta = self.losses[i] - self.losses[i - 1];
            sum += delta.abs() / self.losses[i - 1];
        }

        let s = sum / (*self.winsize - 1) as f64;
        s <= self.threshold
    }
}
