use pyo3::prelude::*;

/// Barrier synchronization — all workers sync at each epoch.
///
/// # Args
/// This constructor does not take arguments.
///
/// # Returns
/// A barrier synchronization configuration.
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
///
/// # Args
/// This constructor does not take arguments.
///
/// # Returns
/// A non-blocking synchronization configuration.
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
