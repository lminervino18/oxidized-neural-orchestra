use std::{error::Error, fmt, io};

use machine_learning::error::MlErr;

pub type Result<T> = std::result::Result<T, WorkerErr>;

/// Worker runtime failures.
#[derive(Debug)]
pub enum WorkerErr {
    Io(io::Error),
    Ml(MlErr),
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
        let s = match self {
            WorkerErr::Io(e) => format!("io error: {e}"),
            WorkerErr::Ml(e) => {
                format!("ml error: {e}")
            }
            WorkerErr::UnexpectedMessage { epoch, got } => {
                format!("unexpected message at step {epoch}: got {got}")
            }
            WorkerErr::WeightsLengthMismatch {
                epoch,
                got,
                expected,
            } => format!("weights length mismatch at step {epoch}: got {got}, expected {expected}"),
            WorkerErr::GradientLengthMismatch {
                epoch,
                got,
                expected,
            } => {
                format!("gradient length mismatch at step {epoch}: got {got}, expected {expected}")
            }
        };

        write!(f, "{s}")
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

impl From<MlErr> for WorkerErr {
    fn from(value: MlErr) -> Self {
        Self::Ml(value)
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
