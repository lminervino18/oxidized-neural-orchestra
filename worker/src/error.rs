use std::{error::Error, fmt, io};

/// The worker module's result type.
pub type Result<T> = std::result::Result<T, WorkerErr>;

/// Worker runtime failures.
#[derive(Debug)]
pub enum WorkerErr {
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

impl fmt::Display for WorkerErr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WorkerErr::Io(e) => write!(f, "io error: {e}"),
            WorkerErr::UnexpectedMessage { epoch, got } => {
                write!(f, "unexpected message at step {epoch}: got {got}")
            }
            WorkerErr::WeightsLengthMismatch {
                epoch,
                got,
                expected,
            } => write!(
                f,
                "weights length mismatch at step {epoch}: got {got}, expected {expected}"
            ),
            WorkerErr::GradientLengthMismatch {
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

impl Error for WorkerErr {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            WorkerErr::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for WorkerErr {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

/// Boundary conversion for binaries / I/O APIs.
impl From<WorkerErr> for io::Error {
    fn from(value: WorkerErr) -> Self {
        match value {
            WorkerErr::Io(e) => e,
            other => io::Error::new(io::ErrorKind::InvalidData, other),
        }
    }
}
