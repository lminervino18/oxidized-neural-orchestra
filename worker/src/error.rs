use std::{error::Error, fmt, io};

/// Worker runtime failures.
#[derive(Debug)]
pub enum WorkerError {
    Io(io::Error),
    UnexpectedMessage {
        step: usize,
        got: &'static str,
    },
    WeightsLengthMismatch {
        step: usize,
        got: usize,
        expected: usize,
    },
    GradientLengthMismatch {
        step: usize,
        got: usize,
        expected: usize,
    },
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
            WorkerError::GradientLengthMismatch {
                step,
                got,
                expected,
            } => write!(
                f,
                "gradient length mismatch at step {step}: got {got}, expected {expected}"
            ),
        }
    }
}

impl Error for WorkerError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            WorkerError::Io(e) => Some(e),
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
            other => io::Error::new(io::ErrorKind::InvalidData, other),
        }
    }
}
