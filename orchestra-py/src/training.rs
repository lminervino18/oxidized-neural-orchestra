use std::num::NonZeroUsize;

use orchestrator::{
    configs::{
        AlgorithmConfig, DatasetConfig, DatasetSrc, LossFnConfig, OptimizerConfig, StoreConfig,
        SynchronizerConfig, TrainingConfig,
    },
    train,
};
use pyo3::prelude::*;

use crate::{
    arch::Sequential,
    datasets::{InlineDataset, LocalDataset},
    optimizers::GradientDescent,
    session::Session,
    store::{BlockingStore, WildStore},
    sync::{BarrierSync, NonBlockingSync},
};

/// Opaque training configuration produced by `parameter_server(...)`.
#[pyclass]
pub struct PyTrainingConfig {
    pub inner: TrainingConfig<String>,
    pub max_epochs: usize,
    pub worker_count: usize,
}

/// Builds a Parameter Server training configuration.
///
/// # Args
/// * `worker_addrs` - List of worker addresses (e.g. `["127.0.0.1:50000"]`).
/// * `server_addrs` - List of parameter server addresses.
/// * `dataset` - The dataset to train on. Accepts either an `InlineDataset` or a `LocalDataset`.
/// * `optimizer` - The optimizer to use.
/// * `sync` - Synchronization strategy (`BarrierSync()` or `NonBlockingSync()`).
/// * `store` - Parameter store strategy (`BlockingStore()` or `WildStore()`).
/// * `max_epochs` - Maximum number of training epochs.
/// * `batch_size` - Mini-batch size.
/// * `offline_epochs` - Extra local epochs per sync round. Defaults to `0`.
/// * `seed` - Optional random seed for reproducibility.
///
/// # Errors
/// Raises a `ValueError` if required fields are invalid.
/// Raises a `TypeError` if `dataset` is not an `InlineDataset` or `LocalDataset`.
#[pyfunction]
#[pyo3(signature = (
    worker_addrs,
    server_addrs,
    dataset,
    optimizer,
    sync,
    store,
    max_epochs,
    batch_size,
    offline_epochs = 0,
    seed = None,
))]
pub fn parameter_server(
    worker_addrs: Vec<String>,
    server_addrs: Vec<String>,
    dataset: &Bound<'_, PyAny>,
    optimizer: PyRef<GradientDescent>,
    sync: &Bound<'_, PyAny>,
    store: &Bound<'_, PyAny>,
    max_epochs: usize,
    batch_size: usize,
    offline_epochs: usize,
    seed: Option<u64>,
) -> PyResult<PyTrainingConfig> {
    let max_epochs_nz = NonZeroUsize::new(max_epochs).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("max_epochs must be greater than 0")
    })?;
    let batch_size_nz = NonZeroUsize::new(batch_size).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("batch_size must be greater than 0")
    })?;

    let synchronizer = if sync.is_instance_of::<BarrierSync>() {
        SynchronizerConfig::Barrier
    } else if sync.is_instance_of::<NonBlockingSync>() {
        SynchronizerConfig::NonBlocking
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "sync must be BarrierSync() or NonBlockingSync()",
        ));
    };

    let store_cfg = if store.is_instance_of::<BlockingStore>() {
        StoreConfig::Blocking
    } else if store.is_instance_of::<WildStore>() {
        StoreConfig::Wild
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "store must be BlockingStore() or WildStore()",
        ));
    };

    // Resolve dataset config from either InlineDataset or LocalDataset.
    let dataset_config = if let Ok(d) = dataset.extract::<PyRef<InlineDataset>>() {
        DatasetConfig {
            src: DatasetSrc::Inline {
                data: d.data.clone(),
            },
            x_size: d.x_size,
            y_size: d.y_size,
        }
    } else if let Ok(d) = dataset.extract::<PyRef<LocalDataset>>() {
        DatasetConfig {
            src: DatasetSrc::Local {
                path: d.path.clone(),
            },
            x_size: d.x_size,
            y_size: d.y_size,
        }
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "dataset must be an InlineDataset or LocalDataset",
        ));
    };

    let worker_count = worker_addrs.len();

    Ok(PyTrainingConfig {
        inner: TrainingConfig {
            worker_addrs,
            algorithm: AlgorithmConfig::ParameterServer {
                server_addrs,
                synchronizer,
                store: store_cfg,
            },
            dataset: dataset_config,
            optimizer: OptimizerConfig::GradientDescent { lr: optimizer.lr },
            loss_fn: LossFnConfig::Mse,
            batch_size: batch_size_nz,
            max_epochs: max_epochs_nz,
            offline_epochs,
            seed,
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
        .allow_threads(|| {
            std::thread::spawn(move || train(model, training).map_err(|e| e.to_string()))
                .join()
                .map_err(|_| "thread panicked".to_string())?
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    Ok(Session {
        inner: Some(session),
        max_epochs,
        worker_count,
    })
}