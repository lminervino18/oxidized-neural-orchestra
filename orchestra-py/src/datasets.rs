use std::num::NonZeroUsize;
use std::path::PathBuf;

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

/// A dataset loaded from a local binary file of packed `f32` values.
///
/// The file must contain raw little-endian `f32` values in row-major order,
/// where each row is `x_size + y_size` floats.
///
/// # Args
/// * `path` - Path to the binary dataset file.
/// * `x_size` - Number of input features per sample.
/// * `y_size` - Number of output features per sample.
#[pyclass]
#[derive(Clone)]
pub struct LocalDataset {
    pub path: PathBuf,
    pub x_size: NonZeroUsize,
    pub y_size: NonZeroUsize,
}

#[pymethods]
impl LocalDataset {
    #[new]
    pub fn new(path: String, x_size: usize, y_size: usize) -> PyResult<Self> {
        let x_size = NonZeroUsize::new(x_size)
            .ok_or_else(|| PyValueError::new_err("x_size must be greater than 0"))?;
        let y_size = NonZeroUsize::new(y_size)
            .ok_or_else(|| PyValueError::new_err("y_size must be greater than 0"))?;
        let path = PathBuf::from(path);
        if !path.exists() {
            return Err(PyValueError::new_err(format!(
                "dataset file not found: {}",
                path.display()
            )));
        }
        Ok(Self {
            path,
            x_size,
            y_size,
        })
    }
}