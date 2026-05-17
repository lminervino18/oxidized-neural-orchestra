use std::{
    error::Error,
    fmt::{self, Display},
    panic::Location,
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
        location: &'static Location<'static>,
    },
    MatrixError {
        source: ndarray::ShapeError,
        location: &'static Location<'static>,
    },
    Conv3dError {
        source: ndarray_conv::Error<3>,
        location: &'static Location<'static>,
    },
    Conv2dError {
        source: ndarray_conv::Error<2>,
        location: &'static Location<'static>,
    },
}

impl MlErr {
    #[track_caller]
    pub fn size_mismatch(what: &'static str, got: usize, expected: usize) -> MlErr {
        MlErr::SizeMismatch {
            what,
            got,
            expected,
            location: Location::caller(),
        }
    }

    #[track_caller]
    pub fn matrix_error(source: ndarray::ShapeError) -> MlErr {
        MlErr::MatrixError {
            source,
            location: Location::caller(),
        }
    }
}

impl Display for MlErr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            MlErr::SizeMismatch {
                what,
                got,
                expected,
                location,
            } => format!("size mismatch for {what}: got {got}, expected {expected} at {location}"),
            MlErr::MatrixError { source, location } => {
                format!("matrix operation failed: {source} at {location}")
            }
            MlErr::Conv3dError { source, location } => {
                format!("convolution operation failed: {source} at {location}")
            }
            MlErr::Conv2dError { source, location } => {
                format!("convolution operation failed: {source} at {location}")
            }
        };

        write!(f, "{s}")
    }
}

impl Error for MlErr {}

impl From<ndarray::ShapeError> for MlErr {
    fn from(value: ndarray::ShapeError) -> Self {
        MlErr::MatrixError {
            source: value,
            location: Location::caller(),
        }
    }
}

impl From<ndarray_conv::Error<3>> for MlErr {
    fn from(value: ndarray_conv::Error<3>) -> Self {
        MlErr::Conv3dError {
            source: value,
            location: Location::caller(),
        }
    }
}

impl From<ndarray_conv::Error<2>> for MlErr {
    fn from(value: ndarray_conv::Error<2>) -> Self {
        MlErr::Conv2dError {
            source: value,
            location: Location::caller(),
        }
    }
}
