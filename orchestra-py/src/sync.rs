use pyo3::prelude::*;

/// Barrier synchronization — all workers sync at each epoch.
#[pyclass]
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
#[pyclass]
#[derive(Clone)]
pub struct NonBlockingSync;

#[pymethods]
impl NonBlockingSync {
    #[new]
    pub fn new() -> Self {
        Self
    }
}
