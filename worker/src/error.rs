use std::{error::Error, fmt, io};

/// Worker runtime failures.
#[derive(Debug)]
pub enum WorkerError {
    Io(io::Error),
    UnexpectedMessage {
        epoch: usize,
        got: &'static str,
    },
    WeightsLengthMismatch {
        epoch: usize,
        got: usize,
        expected: usize,
    },
    GradientLengthMismatch {
        epoch: usize,
        got: usize,
        expected: usize,
    },
}

impl fmt::Display for WorkerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WorkerError::Io(e) => write!(f, "io error: {e}"),
            WorkerError::UnexpectedMessage { epoch, got } => {
                write!(f, "unexpected message at step {epoch}: got {got}")
            }
            WorkerError::WeightsLengthMismatch {
                epoch,
                got,
                expected,
            } => write!(
                f,
                "weights length mismatch at step {epoch}: got {got}, expected {expected}"
            ),
            WorkerError::GradientLengthMismatch {
                epoch,
                got,
                expected,
            } => write!(
                f,
                "gradient length mismatch at step {epoch}: got {got}, expected {expected}"
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

/// Boundary conversion for binaries / I/O APIs.
impl From<WorkerError> for io::Error {
    fn from(value: WorkerError) -> Self {
        match value {
            WorkerError::Io(e) => e,
            other => io::Error::new(io::ErrorKind::InvalidData, other),
        }
    }
}
