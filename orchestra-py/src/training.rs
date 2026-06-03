use std::thread;

use orchestrator::{
    configs::{AlgorithmConfig, TrainingConfig},
    train, CancelHandle,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::{
    arch::Sequential,
    convert::{
        extract_dataset, extract_early_stopping, extract_loss_fn, extract_optimizer,
        extract_serializer, extract_store, extract_synchronizer, parse_nonzero,
    },
    session::Session,
};

/// Opaque training configuration produced by `parameter_server(...)`, `all_reduce(...)`,
/// or `strategy_switch(...)`.
#[pyclass]
pub struct PyTrainingConfig {
    pub inner: TrainingConfig,
    pub max_epochs: usize,
    pub worker_count: usize,
}

/// Builds a Parameter Server training configuration.
///
/// # Args
/// * `addrs` - List of node network addresses (e.g. `["127.0.0.1:50000"]`).
/// * `nservers` - The amount of server nodes (must be lower than the amount of given addresses).
/// * `dataset` - The dataset to train on. Accepts either an `InlineDataset` or a `LocalDataset`.
/// * `optimizer` - The optimizer to use.
/// * `loss_fn` - The loss function to use. Accepts either `Mse()` or `CrossEntropy()`.
/// * `sync` - Synchronization strategy (`BarrierSync()` or `NonBlockingSync()`).
/// * `store` - Parameter store strategy (`BlockingStore()` or `WildStore()`).
/// * `max_epochs` - Maximum number of training epochs.
/// * `batch_size` - Mini-batch size.
/// * `serializer` - Gradient serializer strategy. Accepts either `BaseSerializer()` or `SparseSerializer(r=...)`. Defaults to `BaseSerializer()`.
/// * `offline_epochs` - Extra local epochs per sync round. Defaults to `0`.
/// * `seed` - Optional random seed for reproducibility.
/// * `early_stopping_tolerance` - If set, training stops when the absolute loss improvement between sync rounds is below this value. Must be > 0. Defaults to `None`.
///
/// # Returns
/// A `PyTrainingConfig` ready to be passed to `orchestrate(...)`.
///
/// # Errors
/// Raises a `ValueError` if required fields are invalid.
/// Raises a `TypeError` if any argument has an unsupported type.
#[pyfunction]
#[pyo3(signature = (
    addrs,
    nservers,
    dataset,
    optimizer,
    loss_fn,
    sync,
    store,
    max_epochs,
    batch_size,
    serializer = None,
    offline_epochs = 0,
    seed = None,
    early_stopping_tolerance = None,
))]
pub fn parameter_server(
    addrs: Vec<String>,
    nservers: usize,
    dataset: &Bound<'_, PyAny>,
    optimizer: &Bound<'_, PyAny>,
    loss_fn: &Bound<'_, PyAny>,
    sync: &Bound<'_, PyAny>,
    store: &Bound<'_, PyAny>,
    max_epochs: usize,
    batch_size: usize,
    serializer: Option<&Bound<'_, PyAny>>,
    offline_epochs: usize,
    seed: Option<u64>,
    early_stopping_tolerance: Option<f64>,
) -> PyResult<PyTrainingConfig> {
    let nservers_nz = parse_nonzero(nservers, "nservers")?;
    let max_epochs_nz = parse_nonzero(max_epochs, "max_epochs")?;
    let batch_size_nz = parse_nonzero(batch_size, "batch_size")?;
    let worker_count = addrs.len() - nservers;

    Ok(PyTrainingConfig {
        inner: TrainingConfig {
            addrs,
            algorithm: AlgorithmConfig::ParameterServer {
                nservers: nservers_nz,
                synchronizer: extract_synchronizer(sync)?,
                store: extract_store(store)?,
            },
            serializer: extract_serializer(serializer)?,
            dataset: extract_dataset(dataset)?,
            optimizer: extract_optimizer(optimizer)?,
            loss_fn: extract_loss_fn(loss_fn)?,
            batch_size: batch_size_nz,
            max_epochs: max_epochs_nz,
            offline_epochs,
            seed,
            early_stopping: extract_early_stopping(early_stopping_tolerance)?,
        },
        max_epochs,
        worker_count,
    })
}

/// Builds an All-Reduce training configuration.
///
/// # Args
/// * `addrs` - List of node network addresses (e.g. `["127.0.0.1:50000"]`).
/// * `dataset` - The dataset to train on. Accepts either an `InlineDataset` or a `LocalDataset`.
/// * `optimizer` - The optimizer to use.
/// * `loss_fn` - The loss function to use. Accepts either `Mse()` or `CrossEntropy()`.
/// * `max_epochs` - Maximum number of training epochs.
/// * `batch_size` - Mini-batch size.
/// * `serializer` - Gradient serializer strategy. Accepts either `BaseSerializer()` or `SparseSerializer(r=...)`. Defaults to `BaseSerializer()`.
/// * `offline_epochs` - Extra local epochs per sync round. Defaults to `0`.
/// * `seed` - Optional random seed for reproducibility.
/// * `early_stopping_tolerance` - If set, training stops when the absolute loss improvement between sync rounds is below this value. Must be > 0. Defaults to `None`.
///
/// # Returns
/// A `PyTrainingConfig` ready to be passed to `orchestrate(...)`.
///
/// # Errors
/// Raises a `ValueError` if required fields are invalid.
/// Raises a `TypeError` if any argument has an unsupported type.
#[pyfunction]
#[pyo3(signature = (
    addrs,
    dataset,
    optimizer,
    loss_fn,
    max_epochs,
    batch_size,
    serializer = None,
    offline_epochs = 0,
    seed = None,
    early_stopping_tolerance = None,
))]
pub fn all_reduce(
    addrs: Vec<String>,
    dataset: &Bound<'_, PyAny>,
    optimizer: &Bound<'_, PyAny>,
    loss_fn: &Bound<'_, PyAny>,
    max_epochs: usize,
    batch_size: usize,
    serializer: Option<&Bound<'_, PyAny>>,
    offline_epochs: usize,
    seed: Option<u64>,
    early_stopping_tolerance: Option<f64>,
) -> PyResult<PyTrainingConfig> {
    let max_epochs_nz = parse_nonzero(max_epochs, "max_epochs")?;
    let batch_size_nz = parse_nonzero(batch_size, "batch_size")?;
    let worker_count = addrs.len();

    Ok(PyTrainingConfig {
        inner: TrainingConfig {
            addrs,
            algorithm: AlgorithmConfig::AllReduce,
            serializer: extract_serializer(serializer)?,
            dataset: extract_dataset(dataset)?,
            optimizer: extract_optimizer(optimizer)?,
            loss_fn: extract_loss_fn(loss_fn)?,
            batch_size: batch_size_nz,
            max_epochs: max_epochs_nz,
            offline_epochs,
            seed,
            early_stopping: extract_early_stopping(early_stopping_tolerance)?,
        },
        max_epochs,
        worker_count,
    })
}

/// Builds a Strategy Switch training configuration.
///
/// Starts with AllReduce (all nodes participate) and switches to Parameter Server
/// once the training's relative loss improvement drops below the internal threshold.
///
/// # Args
/// * `addrs` - List of node network addresses (e.g. `["127.0.0.1:50000"]`).
/// * `nservers` - The amount of server nodes (must be lower than the amount of given addresses).
/// * `dataset` - The dataset to train on. Accepts either an `InlineDataset` or a `LocalDataset`.
/// * `optimizer` - The optimizer to use.
/// * `loss_fn` - The loss function to use. Accepts either `Mse()` or `CrossEntropy()`.
/// * `sync` - Synchronization strategy for the PS phase (`BarrierSync()` or `NonBlockingSync()`).
/// * `store` - Parameter store strategy for the PS phase (`BlockingStore()` or `WildStore()`).
/// * `max_epochs` - Maximum number of training epochs.
/// * `batch_size` - Mini-batch size.
/// * `serializer` - Gradient serializer strategy. Accepts either `BaseSerializer()` or `SparseSerializer(r=...)`. Defaults to `BaseSerializer()`.
/// * `offline_epochs` - Extra local epochs per sync round. Defaults to `0`.
/// * `seed` - Optional random seed for reproducibility.
/// * `early_stopping_tolerance` - If set, training stops when the loss improvement is below this value. Defaults to `None`.
///
/// # Returns
/// A `PyTrainingConfig` ready to be passed to `orchestrate(...)`.
///
/// # Errors
/// Raises a `ValueError` if required fields are invalid.
/// Raises a `TypeError` if any argument has an unsupported type.
#[pyfunction]
#[pyo3(signature = (
    addrs,
    nservers,
    dataset,
    optimizer,
    loss_fn,
    sync,
    store,
    max_epochs,
    batch_size,
    serializer = None,
    offline_epochs = 0,
    seed = None,
    early_stopping_tolerance = None,
))]
pub fn strategy_switch(
    addrs: Vec<String>,
    nservers: usize,
    dataset: &Bound<'_, PyAny>,
    optimizer: &Bound<'_, PyAny>,
    loss_fn: &Bound<'_, PyAny>,
    sync: &Bound<'_, PyAny>,
    store: &Bound<'_, PyAny>,
    max_epochs: usize,
    batch_size: usize,
    serializer: Option<&Bound<'_, PyAny>>,
    offline_epochs: usize,
    seed: Option<u64>,
    early_stopping_tolerance: Option<f64>,
) -> PyResult<PyTrainingConfig> {
    let nservers_nz = parse_nonzero(nservers, "nservers")?;
    let max_epochs_nz = parse_nonzero(max_epochs, "max_epochs")?;
    let batch_size_nz = parse_nonzero(batch_size, "batch_size")?;
    let worker_count = addrs.len();

    Ok(PyTrainingConfig {
        inner: TrainingConfig {
            addrs,
            algorithm: AlgorithmConfig::StrategySwitch {
                nservers: nservers_nz,
                synchronizer: extract_synchronizer(sync)?,
                store: extract_store(store)?,
            },
            serializer: extract_serializer(serializer)?,
            dataset: extract_dataset(dataset)?,
            optimizer: extract_optimizer(optimizer)?,
            loss_fn: extract_loss_fn(loss_fn)?,
            batch_size: batch_size_nz,
            max_epochs: max_epochs_nz,
            offline_epochs,
            seed,
            early_stopping: extract_early_stopping(early_stopping_tolerance)?,
        },
        max_epochs,
        worker_count,
    })
}

/// Starts a distributed training session and returns a `Session` handle.
///
/// # Args
/// * `model` - The model to train.
/// * `training` - The training configuration produced by `parameter_server(...)`.
///
/// # Returns
/// A `Session` handle that can be consumed with `wait()`.
///
/// # Errors
/// Raises a `RuntimeError` if the session cannot be started.
#[pyfunction]
pub fn orchestrate(
    py: Python<'_>,
    model: &Sequential,
    training: &PyTrainingConfig,
) -> PyResult<Session> {
    let model = model.inner.clone();
    let max_epochs = training.max_epochs;
    let worker_count = training.worker_count;
    let training = training.inner.clone();

    let session = py
        .detach(|| {
            thread::spawn(move || train(model, training).map_err(|e| e.to_string()))
                .join()
                .map_err(|_| "thread panicked".to_string())?
        })
        .map_err(PyRuntimeError::new_err)?;

    let (cancel, cancel_rx) = CancelHandle::pair();

    Ok(Session {
        inner: Some((session, cancel_rx)),
        cancel,
        max_epochs,
        worker_count,
    })
}
