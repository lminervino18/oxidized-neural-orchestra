use std::fmt;

/// Errors produced while accessing dataset samples.
#[derive(Debug)]
pub enum DataError {
    /// The requested sample index is out of bounds.
    OutOfBounds { index: usize },

    /// The dataset could not provide a valid sample due to domain constraints.
    InvalidSample(&'static str),
}

impl fmt::Display for DataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataError::OutOfBounds { index } => write!(f, "sample index {index} is out of bounds"),
            DataError::InvalidSample(msg) => write!(f, "invalid sample: {msg}"),
        }
    }
}

impl std::error::Error for DataError {}

/// A collection of samples that can be consumed by training strategies.
///
/// A `Dataset` is responsible only for *providing access* to samples.
/// It does not define:
/// - how samples are interpreted,
/// - how they are batched,
/// - whether they contain labels,
/// - any specific model or loss function.
pub trait Dataset: Send {
    /// Sample type produced by this dataset.
    type Sample: Send + Sync;

    /// Returns the total number of samples if known.
    ///
    /// Streaming or infinite datasets should return `None`.
    fn len(&self) -> Option<usize> {
        None
    }

    /// Fetches a sample by index.
    ///
    /// This method is only required to be valid when `len()` returns `Some`.
    ///
    /// # Errors
    /// Returns `DataError::OutOfBounds` if `index` is invalid.
    fn get(&self, index: usize) -> Result<Self::Sample, DataError>;
}
