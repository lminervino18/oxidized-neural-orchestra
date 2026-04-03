use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Base serializer — gradients are always sent densely.
#[pyclass]
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
#[pyclass]
#[derive(Clone)]
pub struct SparseSerializer {
    pub r: f32,
}

#[pymethods]
impl SparseSerializer {
    #[new]
    pub fn new(r: f32) -> PyResult<Self> {
        if !(0.0..=1.0).contains(&r) {
            return Err(PyValueError::new_err("r must be between 0.0 and 1.0"));
        }

        Ok(Self { r })
    }
}
