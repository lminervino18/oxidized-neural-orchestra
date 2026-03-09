use pyo3::prelude::*;

/// Gradient descent optimizer.
///
/// # Args
/// * `lr` - Learning rate.
#[pyclass]
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