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

/// Hyperbolic tangent activation function.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct Tanh {
    pub amp: f32,
}

#[pymethods]
impl Tanh {
    /// Creates a new tanh activation configuration.
    ///
    /// # Args
    /// * `amp` - Amplitude of the tanh. Defaults to `1.0`.
    ///
    /// # Returns
    /// A tanh activation configuration.
    #[new]
    #[pyo3(signature = (amp = 1.0))]
    pub fn new(amp: f32) -> Self {
        Self { amp }
    }
}

/// Rectified Linear Unit activation function.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct ReLU;

#[pymethods]
impl ReLU {
    /// Creates a new relu activation configuration.
    ///
    /// # Returns
    /// A relu activation configuration.
    #[new]
    pub fn new() -> Self {
        Self
    }
}
