use pyo3::prelude::*;

/// Mean squared error loss function.
///
/// # Args
/// This constructor does not take arguments.
///
/// # Returns
/// An MSE loss configuration.
///
/// # Errors
/// This constructor does not return errors.
///
/// # Panics
/// This constructor does not panic.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct Mse;

#[pymethods]
impl Mse {
    #[new]
    pub fn new() -> Self {
        Self
    }
}

/// Cross entropy loss function.
///
/// # Args
/// This constructor does not take arguments.
///
/// # Returns
/// A cross entropy loss configuration.
///
/// # Errors
/// This constructor does not return errors.
///
/// # Panics
/// This constructor does not panic.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct CrossEntropy;

#[pymethods]
impl CrossEntropy {
    #[new]
    pub fn new() -> Self {
        Self
    }
}
