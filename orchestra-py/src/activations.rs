use pyo3::prelude::*;

/// Sigmoid activation function.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct Sigmoid {
    pub amp: f32,
}

#[pymethods]
impl Sigmoid {
    /// Creates a sigmoid activation configuration.
    ///
    /// # Args
    /// * `amp` - Amplitude of the sigmoid. Defaults to `1.0`.
    ///
    /// # Returns
    /// A sigmoid activation configuration.
    #[new]
    #[pyo3(signature = (amp = 1.0))]
    pub fn new(amp: f32) -> Self {
        Self { amp }
    }
}

/// Softmax activation function (row-wise normalization).
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct Softmax;

#[pymethods]
impl Softmax {
    /// Creates a softmax activation configuration.
    ///
    /// # Returns
    /// A softmax activation configuration.
    #[new]
    pub fn new() -> Self {
        Self
    }
}
