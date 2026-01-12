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
            MlError::ShapeMismatch { what, got, expected } => {
                write!(f, "shape mismatch for {what}: got {got}, expected {expected}")
            }
        }
    }
}

impl std::error::Error for MlError {}
