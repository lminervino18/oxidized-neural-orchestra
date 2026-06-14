use pyo3::prelude::*;

/// Gradient descent optimizer.
///
/// # Args
/// * `lr` - Learning rate.
///
/// # Returns
/// A gradient descent optimizer configuration.
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

/// Gradient descent optimizer with momentum.
///
/// # Args
/// * `lr` - Learning rate.
/// * `mu` - Momentum coefficient, between `0.0` and `1.0`.
///
/// # Returns
/// A gradient descent with momentum optimizer configuration.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct GradientDescentWithMomentum {
    pub lr: f32,
    pub mu: f32,
}

#[pymethods]
impl GradientDescentWithMomentum {
    #[new]
    pub fn new(lr: f32, mu: f32) -> Self {
        Self { lr, mu }
    }
}

/// Adam optimizer.
///
/// # Args
/// * `lr` - Learning rate.
/// * `b1` - First moment decay rate, between `0.0` and `1.0`. Defaults to `0.9`.
/// * `b2` - Second moment decay rate, between `0.0` and `1.0`. Defaults to `0.999`.
/// * `eps` - Term added for numerical stability. Defaults to `1e-8`.
///
/// # Returns
/// An adam optimizer configuration.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct Adam {
    pub lr: f32,
    pub b1: f32,
    pub b2: f32,
    pub eps: f32,
}

#[pymethods]
impl Adam {
    #[new]
    #[pyo3(signature = (lr, b1 = 0.9, b2 = 0.999, eps = 1e-8))]
    pub fn new(lr: f32, b1: f32, b2: f32, eps: f32) -> Self {
        Self { lr, b1, b2, eps }
    }
}
