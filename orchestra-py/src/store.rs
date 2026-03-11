use pyo3::prelude::*;

/// Blocking parameter store — gradient updates are applied synchronously.
#[pyclass]
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
#[pyclass]
#[derive(Clone)]
pub struct WildStore;

#[pymethods]
impl WildStore {
    #[new]
    pub fn new() -> Self {
        Self
    }
}
