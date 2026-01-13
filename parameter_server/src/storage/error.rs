use std::{
    error::Error,
    fmt::{self, Display},
};

/// Error returned by `pull_weights` methods when there is a mismatch between the given
/// output slice and the amount of parameters that the `ParameterShard` holds.
#[derive(Debug)]
pub struct SizeMismatchErr;

impl Display for SizeMismatchErr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ParameterShard error: the provided buffer length doesn't match the shard size")
    }
}

impl Error for SizeMismatchErr {}
