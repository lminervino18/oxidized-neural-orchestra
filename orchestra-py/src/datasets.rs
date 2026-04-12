use std::num::NonZeroUsize;
use std::path::PathBuf;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// An inline dataset defined directly in Python.
///
/// # Args
/// * `samples` - Flat list of sample floats in row-major order.
/// * `labels` - Flat list of label floats in row-major order.
/// * `x_size` - Number of input features per sample.
/// * `y_size` - Number of output features per sample.
///
/// # Returns
/// An inline dataset configuration.
///
/// # Errors
/// Raises a `ValueError` if `x_size` is zero.
/// Raises a `ValueError` if `y_size` is zero.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct InlineDataset {
    pub samples: Vec<f32>,
    pub labels: Vec<f32>,
    pub x_size: NonZeroUsize,
    pub y_size: NonZeroUsize,
}

#[pymethods]
impl InlineDataset {
    #[new]
    pub fn new(
        samples: Vec<f32>,
        labels: Vec<f32>,
        x_size: usize,
        y_size: usize,
    ) -> PyResult<Self> {
        let x_size = NonZeroUsize::new(x_size)
            .ok_or_else(|| PyValueError::new_err("x_size must be greater than 0"))?;
        let y_size = NonZeroUsize::new(y_size)
            .ok_or_else(|| PyValueError::new_err("y_size must be greater than 0"))?;
        Ok(Self {
            samples,
            labels,
            x_size,
            y_size,
        })
    }
}

/// A dataset loaded from local binary files of packed `f32` values.
///
/// The files must contain raw little-endian `f32` values in row-major order.
///
/// # Args
/// * `samples_path` - Path to the binary samples dataset file.
/// * `labels_path` - Path to the binary labels dataset file.
/// * `x_size` - Number of input features per sample.
/// * `y_size` - Number of output features per sample.
///
/// # Returns
/// A local dataset configuration.
///
/// # Errors
/// Raises a `ValueError` if `x_size` is zero.
/// Raises a `ValueError` if `y_size` is zero.
/// Raises a `ValueError` if `path` does not exist.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct LocalDataset {
    pub samples_path: PathBuf,
    pub labels_path: PathBuf,
    pub x_size: NonZeroUsize,
    pub y_size: NonZeroUsize,
}

#[pymethods]
impl LocalDataset {
    #[new]
    pub fn new(
        samples_path: String,
        labels_path: String,
        x_size: usize,
        y_size: usize,
    ) -> PyResult<Self> {
        let x_size = NonZeroUsize::new(x_size)
            .ok_or_else(|| PyValueError::new_err("x_size must be greater than 0"))?;
        let y_size = NonZeroUsize::new(y_size)
            .ok_or_else(|| PyValueError::new_err("y_size must be greater than 0"))?;

        let samples_path = PathBuf::from(samples_path);
        if !samples_path.exists() {
            return Err(PyValueError::new_err(format!(
                "dataset samples file not found: {}",
                samples_path.display()
            )));
        }

        let labels_path = PathBuf::from(labels_path);
        if !labels_path.exists() {
            return Err(PyValueError::new_err(format!(
                "dataset labels file not found: {}",
                labels_path.display()
            )));
        }

        Ok(Self {
            samples_path,
            labels_path,
            x_size,
            y_size,
        })
    }
}
