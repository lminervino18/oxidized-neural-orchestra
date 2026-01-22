use std::{error::Error, fmt, io};

use machine_learning::MlError;

/// Worker runtime failures.
#[derive(Debug)]
pub enum WorkerError {
    Io(io::Error),

    UnexpectedMessage { step: usize, got: &'static str },

    WeightsLengthMismatch {
        step: usize,
        got: usize,
        expected: usize,
    },

    TrainFailed { step: usize, source: MlError },
}

impl fmt::Display for WorkerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WorkerError::Io(e) => write!(f, "io error: {e}"),
            WorkerError::UnexpectedMessage { step, got } => {
                write!(f, "unexpected message at step {step}: got {got}")
            }
            WorkerError::WeightsLengthMismatch {
                step,
                got,
                expected,
            } => write!(
                f,
                "weights length mismatch at step {step}: got {got}, expected {expected}"
            ),
            WorkerError::TrainFailed { step, source } => {
                write!(f, "train failed at step {step}: {source}")
            }
        }
    }
}

impl Error for WorkerError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            WorkerError::Io(e) => Some(e),
            WorkerError::TrainFailed { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl From<io::Error> for WorkerError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

impl WorkerError {
    /// Converts this error into an `io::Error` for boundary APIs.
    pub fn into_io(self) -> io::Error {
        match self {
            WorkerError::Io(e) => e,
            WorkerError::UnexpectedMessage { .. } => io::Error::new(io::ErrorKind::InvalidData, self),
            WorkerError::WeightsLengthMismatch { .. } => {
                io::Error::new(io::ErrorKind::InvalidData, self)
            }
            WorkerError::TrainFailed { .. } => io::Error::new(io::ErrorKind::InvalidData, self),
        }
    }
}
