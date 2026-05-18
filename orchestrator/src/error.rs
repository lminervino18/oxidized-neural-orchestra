use std::{
    error::Error,
    fmt::{self, Display},
    io,
};

use crate::sessions::WorkerRequest;

/// The orchestrator module's error type.
#[derive(Debug)]
pub enum OrchErr {
    InvalidConfig(String),
    Unsupported(String),
    ConnectionFailed { addr: String, source: io::Error },
    WorkerError { id: usize, details: String },
    ServerError(String),
    SafeTensors(safetensors::SafeTensorError),
    InvalidRequest(WorkerRequest),
    Io(io::Error),
}

/// The orchestrator module's result type.
pub type Result<T> = std::result::Result<T, OrchErr>;

impl Display for OrchErr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::InvalidConfig(msg) => format!("invalid config: {msg}"),
            Self::Unsupported(msg) => format!("unsupported: {msg}"),
            Self::ConnectionFailed { addr, source } => {
                format!("failed to reach to {addr}: {source}")
            }
            Self::WorkerError { id, details: msg } => {
                format!("worker {id} error: {msg}")
            }
            Self::InvalidRequest(req) => format!("invalid worker request: {req:?}"),
            Self::SafeTensors(e) => format!("safetensors error: {e}"),
            Self::ServerError(msg) => format!("server error: {msg}"),
            Self::Io(e) => format!("io error: {e}"),
        };

        write!(f, "{s}")
    }
}

impl Error for OrchErr {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::ConnectionFailed { source, .. } => Some(source),
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for OrchErr {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<safetensors::SafeTensorError> for OrchErr {
    fn from(value: safetensors::SafeTensorError) -> Self {
        Self::SafeTensors(value)
    }
}
