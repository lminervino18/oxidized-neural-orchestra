use std::{
    error::Error,
    fmt::{self, Display},
    panic::Location,
};

use ndarray::ShapeError;

/// The result type for the `machine_learning` module.
pub type Result<T> = std::result::Result<T, MlErr>;

/// The error type for the `machine_learning` module.
#[derive(Debug)]
pub enum MlErr {
    SizeMismatch {
        what: &'static str,
        got: usize,
        expected: usize,
        location: &'static Location<'static>,
    },
    MatrixError {
        source: ShapeError,
        location: &'static Location<'static>,
    },
    EmptyEpoch,
}

impl MlErr {
    #[track_caller]
    pub fn size_mismatch(what: &'static str, got: usize, expected: usize) -> MlErr {
        MlErr::SizeMismatch {
            what,
            got,
            expected,
            location: Location::caller(),
        }
    }

    #[track_caller]
    pub fn matrix_error(source: ShapeError) -> MlErr {
        MlErr::MatrixError {
            source,
            location: Location::caller(),
        }
    }
}

impl Display for MlErr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            MlErr::SizeMismatch {
                what,
                got,
                expected,
                location,
            } => format!("size mismatch for {what}: got {got}, expected {expected} at {location}"),
            MlErr::MatrixError { source, location } => {
                format!("matrix operation failed: {source} at {location}")
            }
            MlErr::EmptyEpoch => "this epoch has no batches".to_string(),
        };

        write!(f, "{s}")
    }
}

impl Error for MlErr {}

impl From<ShapeError> for MlErr {
    fn from(value: ShapeError) -> Self {
        MlErr::MatrixError {
            source: value,
            location: Location::caller(),
        }
    }
}
