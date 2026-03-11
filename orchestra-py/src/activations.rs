use pyo3::prelude::*;

/// Sigmoid activation function.
///
/// # Args
/// * `amp` - Amplitude of the sigmoid. Defaults to 1.0.
#[pyclass]
#[derive(Clone)]
pub struct Sigmoid {
    pub amp: f32,
}

#[pymethods]
impl Sigmoid {
    #[new]
    #[pyo3(signature = (amp = 1.0))]
    pub fn new(amp: f32) -> Self {
        Self { amp }
    }
}
