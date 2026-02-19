use std::{
    error::Error,
    fmt::{self, Display},
};

/// The result type used in the entire machine learning module.
pub type Result<T> = std::result::Result<T, MlErr>;

/// The machine learning module's error type.
#[derive(Debug)]
pub enum MlErr {
    SizeMismatch {
        a: &'static str,
        b: &'static str,
        got: usize,
        expected: usize,
    },
    ParamManagerIsNotFull {
        got: usize,
        expected: usize,
    },
    ParamManagerEmptySlice {
        corresponds: usize,
    },
    ParamManagerEmptyOrdering,
    ParamManagerInvalidOrderingIds {
        unique: usize,
        max: usize,
    },
}

impl Display for MlErr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            MlErr::SizeMismatch {
                a,
                b,
                got,
                expected,
            } => {
                format!(
                    "There's a size mismatch between {a} and {b}, got {got} and expected {expected}"
                )
            }
            MlErr::ParamManagerIsNotFull { got, expected } => format!(
                "The parameter manager is not full, it has {got} parameter slices of the expected {expected}"
            ),
            MlErr::ParamManagerEmptySlice { corresponds } => format!(
                "Tried to iterate the parameters when at least the {corresponds}-th slice is missing"
            ),
            MlErr::ParamManagerEmptyOrdering => {
                format!("Failed to instanciate parameter manager, the given ordering vec is empty")
            }
            MlErr::ParamManagerInvalidOrderingIds { unique, max } => format!(
                "The given ordering is invalid, there are {unique} unique values and the maximum value is {max}"
            ),
        };

        write!(f, "{s}")
    }
}

impl Error for MlErr {}
