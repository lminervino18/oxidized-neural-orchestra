use std::fmt;

/// Errors produced by ML plugins when inputs are invalid.
#[derive(Debug)]
pub enum MlError {
    InvalidInput(&'static str),
    ShapeMismatch { what: &'static str, got: usize, expected: usize },
}

impl fmt::Display for MlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MlError::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            MlError::ShapeMismatch { what, got, expected } => {
                write!(f, "shape mismatch for {what}: got {got}, expected {expected}")
            }
        }
    }
}

impl std::error::Error for MlError {}

/// Statistics emitted by one local step.
///
/// Fields are private to avoid leaking internal counters as public API.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StepStats {
    microbatches: usize,
    samples: usize,
}

impl StepStats {
    pub fn new(microbatches: usize, samples: usize) -> Self {
        Self { microbatches, samples }
    }

    pub fn microbatches(&self) -> usize {
        self.microbatches
    }

    pub fn samples(&self) -> usize {
        self.samples
    }
}

/// A local training step executed by a worker.
///
/// Contract:
/// - `weights` is borrowed (must not be copied by the worker).
/// - `grads` is provided by the worker and is already zeroed.
/// - The strategy writes gradients into `grads` and returns step statistics.
pub trait TrainStrategy: Send {
    fn num_params(&self) -> usize;

    fn step(&mut self, weights: &[f32], grads: &mut [f32]) -> Result<StepStats, MlError>;
}
