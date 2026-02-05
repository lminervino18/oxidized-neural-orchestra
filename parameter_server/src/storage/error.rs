use std::{
    error::Error,
    fmt::{self, Display},
};

/// The specific result type for size mismatch checks inside the storage module.
pub type Result<T> = std::result::Result<T, SizeMismatchErr>;

/// Error returned by various methods in the `ParameterShard` whenever there is a size
/// mismatch between different gradients, parameters and external buffers.
#[derive(Debug)]
pub struct SizeMismatchErr;

impl Display for SizeMismatchErr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ParameterShard error: the provided buffer length doesn't match the shard size")
    }
}

impl Error for SizeMismatchErr {}
