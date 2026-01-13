use std::{
    error::Error,
    fmt::{self, Display},
};

use rand_distr::{NormalError, uniform::Error as UniformError};

/// The specific result type for the different instances of `RandWeightGen` generators.
pub type Result<T> = std::result::Result<T, RandErr>;

/// Error returned by the `RandWeightGen` constructors whenever there is an error creating
/// a new instance of the struct, each constructor has it's own constraints given that
/// they use different distributions.
#[derive(Debug)]
pub struct RandErr(String);

impl From<NormalError> for RandErr {
    fn from(value: NormalError) -> Self {
        Self(value.to_string())
    }
}

impl From<UniformError> for RandErr {
    fn from(value: UniformError) -> Self {
        Self(value.to_string())
    }
}

impl Display for RandErr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for RandErr {}
