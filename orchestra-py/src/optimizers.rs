use pyo3::prelude::*;

/// Gradient descent optimizer.
///
/// # Args
/// * `lr` - Learning rate.
///
/// # Returns
/// A gradient descent optimizer configuration.
///
/// # Errors
/// This constructor does not return errors.
///
/// # Panics
/// This constructor does not panic.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct GradientDescent {
    pub lr: f32,
}

#[pymethods]
impl GradientDescent {
    #[new]
    pub fn new(lr: f32) -> Self {
        Self { lr }
    }
}
