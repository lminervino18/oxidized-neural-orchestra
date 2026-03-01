use std::fmt;

/// All errors that can occur in the orchestrator.
#[derive(Debug)]
pub enum OrchestratorError {
    /// Invalid configuration — caught before connecting.
    InvalidConfig(String),
    /// Failed to connect to a worker or server.
    ConnectionFailed {
        addr: String,
        source: std::io::Error,
    },
    /// A worker produced an unrecoverable error during training.
    WorkerError { worker_id: usize, msg: String },
    /// The parameter server produced an unrecoverable error.
    ServerError(String),
    /// An underlying I/O error not covered by the above variants.
    Io(std::io::Error),
}

impl fmt::Display for OrchestratorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
            Self::ConnectionFailed { addr, source } => {
                write!(f, "connection failed to {addr}: {source}")
            }
            Self::WorkerError { worker_id, msg } => {
                write!(f, "worker {worker_id} error: {msg}")
            }
            Self::ServerError(msg) => write!(f, "server error: {msg}"),
            Self::Io(e) => write!(f, "io error: {e}"),
        }
    }
}

impl std::error::Error for OrchestratorError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ConnectionFailed { source, .. } => Some(source),
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for OrchestratorError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
