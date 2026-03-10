use std::num::NonZeroUsize;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// An inline dataset defined directly in Python.
///
/// # Args
/// * `data` - Flat list of floats in row-major order.
/// * `x_size` - Number of input features per sample.
/// * `y_size` - Number of output features per sample.
#[pyclass]
#[derive(Clone)]
pub struct InlineDataset {
    pub data: Vec<f32>,
    pub x_size: NonZeroUsize,
    pub y_size: NonZeroUsize,
}

#[pymethods]
impl InlineDataset {
    #[new]
    pub fn new(data: Vec<f32>, x_size: usize, y_size: usize) -> PyResult<Self> {
        let x_size = NonZeroUsize::new(x_size)
            .ok_or_else(|| PyValueError::new_err("x_size must be greater than 0"))?;
        let y_size = NonZeroUsize::new(y_size)
            .ok_or_else(|| PyValueError::new_err("y_size must be greater than 0"))?;
        Ok(Self {
            data,
            x_size,
            y_size,
        })
    }
}
