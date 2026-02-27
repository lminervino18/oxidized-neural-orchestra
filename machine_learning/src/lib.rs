pub mod arch;
pub mod dataset;
pub mod optimization;
mod test;
pub mod training;

use std::fmt;

/// Errors produced by ML plugins when inputs are invalid.
#[derive(Debug)]
pub enum MlError {
    /// An input is invalid for semantic or domain reasons.
    InvalidInput(&'static str),

    /// A shape invariant was violated (e.g. mismatched lengths).
    ShapeMismatch {
        /// Human-readable context for the mismatch (e.g. "params", "batch").
        what: &'static str,
        /// Observed value.
        got: usize,
        /// Expected value.
        expected: usize,
    },
}

impl fmt::Display for MlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MlError::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            MlError::ShapeMismatch {
                what,
                got,
                expected,
            } => {
                write!(
                    f,
                    "shape mismatch for {what}: got {got}, expected {expected}"
                )
            }
        }
    }
}

impl std::error::Error for MlError {}

/// Statistics produced by a single local training step.
///
/// This type keeps fields private to allow evolving the internal counters
/// without breaking the public API.
#[derive(Debug, Default, PartialEq, Eq)]
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
    pub fn new(microbatches: usize, samples: usize) -> Self {
        Self {
            microbatches,
            samples,
        }
    }

    /// Returns the number of microbatches processed in the last step.
    ///
    /// # Returns
    /// The microbatch count.
    ///
    pub fn microbatches(&self) -> usize {
        self.microbatches
    }

    /// Returns the number of samples processed in the last step.
    ///
    /// # Returns
    /// The sample count.
    ///
    pub fn samples(&self) -> usize {
        self.samples
    }
}

/// Abstraction over a local training computation executed by a worker.
///
/// Implementations encapsulate all model-, data-, and loss-specific logic.
/// The worker treats this trait as a black box that maps weights to gradients.
pub trait TrainStrategy: Send {
    /// Executes one local training step.
    ///
    /// # Args
    /// * `weights` - Read-only slice containing the current model parameters.
    /// * `grads` - Mutable gradient buffer provided by the worker. The worker guarantees
    ///   that this buffer is zeroed before calling `step`.
    ///
    /// # Returns
    /// On success, returns step-level statistics (`StepStats`). On failure, returns an `MlError`.
    ///
    /// # Errors
    /// Implementations should return:
    /// - `MlError::ShapeMismatch` when shapes/lengths do not match expectations.
    /// - `MlError::InvalidInput` for invalid domain inputs.
    fn step(&mut self, weights: &[f32], grads: &mut [f32]) -> Result<StepStats, MlError>;
}
