/// Statistics produced by a single local training step.
///
/// This type keeps fields private to allow evolving the internal counters
/// without breaking the public API.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StepStats {
    microbatches: usize,
    samples: usize,
}

impl StepStats {
    /// Creates a new `StepStats`.
    ///
    /// # Args
    /// * `microbatches` - Number of microbatches processed during the step.
    /// * `samples` - Total number of samples processed during the step.
    ///
    /// # Returns
    /// A `StepStats` instance containing the provided counters.
    ///
    /// # Panics
    /// Never panics.
    pub fn new(microbatches: usize, samples: usize) -> Self {
        Self { microbatches, samples }
    }

    /// Returns the number of microbatches processed in the last step.
    ///
    /// # Returns
    /// The microbatch count.
    ///
    /// # Panics
    /// Never panics.
    pub fn microbatches(&self) -> usize {
        self.microbatches
    }

    /// Returns the number of samples processed in the last step.
    ///
    /// # Returns
    /// The sample count.
    ///
    /// # Panics
    /// Never panics.
    pub fn samples(&self) -> usize {
        self.samples
    }
}
