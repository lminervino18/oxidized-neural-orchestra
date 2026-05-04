use comms::floats::Float01;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Base serializer — gradients are always sent densely.
///
/// # Args
/// This constructor does not take arguments.
///
/// # Returns
/// A dense serializer configuration.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct BaseSerializer;

#[pymethods]
impl BaseSerializer {
    #[new]
    pub fn new() -> Self {
        Self
    }
}

/// Sparse serializer — gradients may be compressed before being sent.
///
/// # Args
/// * `r` - Compression ratio threshold between `0.0` and `1.0`.
///
/// # Returns
/// A sparse serializer configuration.
///
/// # Errors
/// Raises a `ValueError` if `r` is outside `[0.0, 1.0]`.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct SparseSerializer {
    pub r: Float01,
}

#[pymethods]
impl SparseSerializer {
    #[new]
    pub fn new(r: f32) -> PyResult<Self> {
        let r = Float01::new(r)
            .ok_or_else(|| PyValueError::new_err("r must be between 0.0 and 1.0"))?;

        Ok(Self { r })
    }
}
