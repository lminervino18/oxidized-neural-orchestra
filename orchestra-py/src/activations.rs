use pyo3::prelude::*;

/// Sigmoid activation function.
///
/// # Args
/// * `amp` - Amplitude of the sigmoid. Defaults to 1.0.
///
/// # Returns
/// A sigmoid activation configuration.
///
/// # Errors
/// This constructor does not return errors.
///
/// # Panics
/// This constructor does not panic.
#[pyclass(skip_from_py_object)]
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
