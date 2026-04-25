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
    MatrixError(ndarray::ShapeError),
    Conv3dError(ndarray_conv::Error<3>),
    Conv2dError(ndarray_conv::Error<2>),
}

impl Display for MlErr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            MlErr::SizeMismatch {
                what,
                got,
                expected,
            } => format!("size mismatch for {what}: got {got}, expected {expected}"),
            MlErr::MatrixError(shape_error) => format!("matrix operation failed: {shape_error}"),
            MlErr::Conv3dError(conv3d_error) => {
                format!("convolution operation failed: {conv3d_error}")
            }
            MlErr::Conv2dError(conv2d_error) => {
                format!("convolution operation failed: {conv2d_error}")
            }
        };

        write!(f, "{s}")
    }
}

impl Error for MlErr {}

impl From<ndarray::ShapeError> for MlErr {
    fn from(value: ndarray::ShapeError) -> Self {
        MlErr::MatrixError(value)
    }
}

impl From<ndarray_conv::Error<3>> for MlErr {
    fn from(value: ndarray_conv::Error<3>) -> Self {
        MlErr::Conv3dError(value)
    }
}

impl From<ndarray_conv::Error<2>> for MlErr {
    fn from(value: ndarray_conv::Error<2>) -> Self {
        MlErr::Conv2dError(value)
    }
}
