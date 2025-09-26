use std::fmt;

/// Enum to represent different error types in the system
#[derive(Debug)]
pub enum SystemError {
    ConnectionError(String),           // Error when a connection fails
    TaskAssignmentError(String),       // Error when assigning tasks to workers
    CommunicationError(String),        // Error in communication between orchestrator and worker
    InvalidDataError(String),          // Error when receiving or processing invalid data
    TimeoutError(String),              // Error when a timeout occurs
    WorkerError(String),               // Error specific to worker execution
    OrchestratorError(String),         // Error specific to orchestrator behavior
    WorkerTimeoutError(String),        // Timeout error for worker processing
    OrchestratorOverloadError(String), // Error when orchestrator is overloaded
    NetworkFailureError(String),       // Error when there are network issues
}

/// Implementing the `fmt::Display` trait for custom error messages
impl fmt::Display for SystemError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            SystemError::ConnectionError(ref msg) => {
                write!(f, "Connection Error: {}", msg)
            }
            SystemError::TaskAssignmentError(ref msg) => {
                write!(f, "Task Assignment Error: {}", msg)
            }
            SystemError::CommunicationError(ref msg) => {
                write!(f, "Communication Error: {}", msg)
            }
            SystemError::InvalidDataError(ref msg) => {
                write!(f, "Invalid Data Error: {}", msg)
            }
            SystemError::TimeoutError(ref msg) => {
                write!(f, "Timeout Error: {}", msg)
            }
            SystemError::WorkerError(ref msg) => {
                write!(f, "Worker Error: {}", msg)
            }
            SystemError::OrchestratorError(ref msg) => {
                write!(f, "Orchestrator Error: {}", msg)
            }
            SystemError::WorkerTimeoutError(ref msg) => {
                write!(f, "Worker Timeout Error: {}", msg)
            }
            SystemError::OrchestratorOverloadError(ref msg) => {
                write!(f, "Orchestrator Overload Error: {}", msg)
            }
            SystemError::NetworkFailureError(ref msg) => {
                write!(f, "Network Failure Error: {}", msg)
            }
        }
    }
}

/// Result type for custom error handling in the system
pub type Result<T> = std::result::Result<T, SystemError>;

/// A utility function to handle errors with a custom message
pub fn handle_error(error: SystemError) {
    eprintln!("Error occurred: {}", error);
    // Optionally, log this error to a file, or trigger alerts.
}
