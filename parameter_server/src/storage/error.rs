use std::{
    error::Error,
    fmt::{self, Display},
};

use machine_learning::MlErr;

/// The specific result type for size mismatch checks inside the storage module.
pub type Result<T> = std::result::Result<T, ParamServerErr>;

/// Error returned by various methods in the `ParameterShard` whenever there is a size
/// mismatch between different gradients, parameters and external buffers.
#[derive(Debug)]
pub enum ParamServerErr {
    SizeMismatch,
    Other,
}

impl Display for ParamServerErr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::SizeMismatch => "The provided buffer length doesn't match the shard size",
            Self::Other => "Unknown parameter server error",
        };

        f.write_str(s)
    }
}

impl Error for ParamServerErr {}

impl TryInto<ParamServerErr> for MlErr {
    type Error = ParamServerErr;

    fn try_into(self) -> std::result::Result<ParamServerErr, Self::Error> {
        match self {
            Self::SizeMismatch { .. } => Ok(ParamServerErr::SizeMismatch),
            _ => Err(ParamServerErr::Other),
        }
    }
}
