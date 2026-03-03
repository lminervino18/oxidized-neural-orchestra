use std::{
    error::Error,
    fmt::{self, Display},
};

/// The result type for the `machine_learning` module.
pub type Result<T> = std::result::Result<T, MlErr>;

/// The error type for the `machine_learning` module.
#[derive(Debug)]
pub enum MlErr {
    SizeMismatch {
        what: &'static str,
        got: usize,
        expected: usize,
    },
}

impl Display for MlErr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            MlErr::SizeMismatch {
                what,
                got,
                expected,
            } => format!("shape mismatch for {what}: got {got}, expected {expected}"),
        };

        write!(f, "{s}")
    }
}

impl Error for MlErr {}
