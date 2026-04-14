use pyo3::prelude::*;

/// Blocking parameter store — gradient updates are applied synchronously.
///
/// # Args
/// This constructor does not take arguments.
///
/// # Returns
/// A blocking store configuration.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct BlockingStore;

#[pymethods]
impl BlockingStore {
    #[new]
    pub fn new() -> Self {
        Self
    }
}

/// Wild parameter store — gradient updates are applied without locking.
///
/// # Args
/// This constructor does not take arguments.
///
/// # Returns
/// A wild store configuration.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct WildStore;

#[pymethods]
impl WildStore {
    #[new]
    pub fn new() -> Self {
        Self
    }
}
