use pyo3::prelude::*;

/// Barrier synchronization — all workers sync at each epoch.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct BarrierSync;

#[pymethods]
impl BarrierSync {
    #[new]
    pub fn new() -> Self {
        Self
    }
}

/// Non-blocking synchronization — workers proceed without waiting.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct NonBlockingSync;

#[pymethods]
impl NonBlockingSync {
    #[new]
    pub fn new() -> Self {
        Self
    }
}
