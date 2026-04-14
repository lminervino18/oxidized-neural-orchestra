use std::num::NonZeroUsize;

use orchestrator::{
    configs::{
        AlgorithmConfig, DatasetConfig, DatasetSrc, LossFnConfig, OptimizerConfig,
        SerializerConfig, StoreConfig, SynchronizerConfig, TrainingConfig,
    },
    train,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::{
    arch::Sequential,
    datasets::{InlineDataset, LocalDataset},
    loss_fns::{CrossEntropy, Mse},
    optimizers::GradientDescent,
    serializer::{BaseSerializer, SparseSerializer},
    session::Session,
    store::{BlockingStore, WildStore},
    sync::{BarrierSync, NonBlockingSync},
};

/// Opaque training configuration produced by `parameter_server(...)`.
#[pyclass]
pub struct PyTrainingConfig {
    pub inner: TrainingConfig,
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
/// * `loss_fn` - The loss function to use. Accepts either `Mse()` or `CrossEntropy()`.
/// * `sync` - Synchronization strategy (`BarrierSync()` or `NonBlockingSync()`).
/// * `store` - Parameter store strategy (`BlockingStore()` or `WildStore()`).
/// * `max_epochs` - Maximum number of training epochs.
/// * `batch_size` - Mini-batch size.
/// * `serializer` - Gradient serializer strategy. Accepts either `BaseSerializer()` or `SparseSerializer(r=...)`. Defaults to `BaseSerializer()`.
/// * `offline_epochs` - Extra local epochs per sync round. Defaults to `0`.
/// * `seed` - Optional random seed for reproducibility.
///
/// # Returns
/// A `PyTrainingConfig` ready to be passed to `orchestrate(...)`.
///
/// # Errors
/// Raises a `ValueError` if required fields are invalid.
/// Raises a `TypeError` if `dataset` is not an `InlineDataset` or `LocalDataset`.
/// Raises a `TypeError` if `loss_fn` is not `Mse()` or `CrossEntropy()`.
/// Raises a `TypeError` if `serializer` is not `BaseSerializer()` or `SparseSerializer(r=...)`.
#[pyfunction]
#[pyo3(signature = (
    worker_addrs,
    server_addrs,
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
))]
pub fn parameter_server(
    worker_addrs: Vec<String>,
    server_addrs: Vec<String>,
    dataset: &Bound<'_, PyAny>,
    optimizer: PyRef<GradientDescent>,
    loss_fn: &Bound<'_, PyAny>,
    sync: &Bound<'_, PyAny>,
    store: &Bound<'_, PyAny>,
    max_epochs: usize,
    batch_size: usize,
    serializer: Option<&Bound<'_, PyAny>>,
    offline_epochs: usize,
    seed: Option<u64>,
) -> PyResult<PyTrainingConfig> {
    let max_epochs_nz = NonZeroUsize::new(max_epochs)
        .ok_or_else(|| PyValueError::new_err("max_epochs must be greater than 0"))?;

    let batch_size_nz = NonZeroUsize::new(batch_size)
        .ok_or_else(|| PyValueError::new_err("batch_size must be greater than 0"))?;

    let synchronizer = if sync.is_instance_of::<BarrierSync>() {
        SynchronizerConfig::Barrier
    } else if sync.is_instance_of::<NonBlockingSync>() {
        SynchronizerConfig::NonBlocking
    } else {
        return Err(PyTypeError::new_err(
            "sync must be BarrierSync() or NonBlockingSync()",
        ));
    };

    let store_cfg = if store.is_instance_of::<BlockingStore>() {
        StoreConfig::Blocking
    } else if store.is_instance_of::<WildStore>() {
        StoreConfig::Wild
    } else {
        return Err(PyTypeError::new_err(
            "store must be BlockingStore() or WildStore()",
        ));
    };

    let loss_fn_cfg = if loss_fn.is_instance_of::<Mse>() {
        LossFnConfig::Mse
    } else if loss_fn.is_instance_of::<CrossEntropy>() {
        LossFnConfig::CrossEntropy
    } else {
        return Err(PyTypeError::new_err(
            "loss_fn must be Mse() or CrossEntropy()",
        ));
    };

    let serializer_cfg = match serializer {
        None => SerializerConfig::Base,
        Some(serializer) if serializer.is_instance_of::<BaseSerializer>() => SerializerConfig::Base,
        Some(serializer) => {
            if let Ok(sparse) = serializer.extract::<PyRef<SparseSerializer>>() {
                SerializerConfig::SparseCapable { r: sparse.r }
            } else {
                return Err(PyTypeError::new_err(
                    "serializer must be BaseSerializer() or SparseSerializer(r=...)",
                ));
            }
        }
    };

    let dataset_config = if let Ok(d) = dataset.extract::<PyRef<InlineDataset>>() {
        DatasetConfig {
            src: DatasetSrc::Inline {
                samples: d.samples.clone(),
                labels: d.labels.clone(),
            },
            x_size: d.x_size,
            y_size: d.y_size,
        }
    } else if let Ok(d) = dataset.extract::<PyRef<LocalDataset>>() {
        DatasetConfig {
            src: DatasetSrc::Local {
                samples_path: d.samples_path.clone(),
                labels_path: d.labels_path.clone(),
            },
            x_size: d.x_size,
            y_size: d.y_size,
        }
    } else {
        return Err(PyTypeError::new_err(
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
            serializer: serializer_cfg,
            dataset: dataset_config,
            optimizer: OptimizerConfig::GradientDescent { lr: optimizer.lr },
            loss_fn: loss_fn_cfg,
            batch_size: batch_size_nz,
            max_epochs: max_epochs_nz,
            offline_epochs,
            seed,
        },
        max_epochs,
        worker_count,
    })
}

/// Builds an All-Reduce training configuration.
///
/// # Args
/// * `worker_addrs` - List of worker addresses (e.g. `["127.0.0.1:50000"]`).
/// * `dataset` - The dataset to train on. Accepts either an `InlineDataset` or a `LocalDataset`.
/// * `optimizer` - The optimizer to use.
/// * `loss_fn` - The loss function to use. Accepts either `Mse()` or `CrossEntropy()`.
/// * `max_epochs` - Maximum number of training epochs.
/// * `batch_size` - Mini-batch size.
/// * `serializer` - Gradient serializer strategy. Accepts either `BaseSerializer()` or `SparseSerializer(r=...)`. Defaults to `BaseSerializer()`.
/// * `offline_epochs` - Extra local epochs per sync round. Defaults to `0`.
/// * `seed` - Optional random seed for reproducibility.
///
/// # Returns
/// A `PyTrainingConfig` ready to be passed to `orchestrate(...)`.
///
/// # Errors
/// Raises a `ValueError` if required fields are invalid.
/// Raises a `TypeError` if `dataset` is not an `InlineDataset` or `LocalDataset`.
/// Raises a `TypeError` if `loss_fn` is not `Mse()` or `CrossEntropy()`.
/// Raises a `TypeError` if `serializer` is not `BaseSerializer()` or `SparseSerializer(r=...)`.
#[pyfunction]
#[pyo3(signature = (
    worker_addrs,
    dataset,
    optimizer,
    loss_fn,
    max_epochs,
    batch_size,
    serializer = None,
    offline_epochs = 0,
    seed = None,
))]
pub fn all_reduce(
    worker_addrs: Vec<String>,
    dataset: &Bound<'_, PyAny>,
    optimizer: PyRef<GradientDescent>,
    loss_fn: &Bound<'_, PyAny>,
    max_epochs: usize,
    batch_size: usize,
    serializer: Option<&Bound<'_, PyAny>>,
    offline_epochs: usize,
    seed: Option<u64>,
) -> PyResult<PyTrainingConfig> {
    let max_epochs_nz = NonZeroUsize::new(max_epochs)
        .ok_or_else(|| PyValueError::new_err("max_epochs must be greater than 0"))?;

    let batch_size_nz = NonZeroUsize::new(batch_size)
        .ok_or_else(|| PyValueError::new_err("batch_size must be greater than 0"))?;

    let loss_fn_cfg = if loss_fn.is_instance_of::<Mse>() {
        LossFnConfig::Mse
    } else if loss_fn.is_instance_of::<CrossEntropy>() {
        LossFnConfig::CrossEntropy
    } else {
        return Err(PyTypeError::new_err(
            "loss_fn must be Mse() or CrossEntropy()",
        ));
    };

    let serializer_cfg = match serializer {
        None => SerializerConfig::Base,
        Some(serializer) if serializer.is_instance_of::<BaseSerializer>() => SerializerConfig::Base,
        Some(serializer) => {
            if let Ok(sparse) = serializer.extract::<PyRef<SparseSerializer>>() {
                SerializerConfig::SparseCapable { r: sparse.r }
            } else {
                return Err(PyTypeError::new_err(
                    "serializer must be BaseSerializer() or SparseSerializer(r=...)",
                ));
            }
        }
    };

    let dataset_config = if let Ok(d) = dataset.extract::<PyRef<InlineDataset>>() {
        DatasetConfig {
            src: DatasetSrc::Inline {
                samples: d.samples.clone(),
                labels: d.labels.clone(),
            },
            x_size: d.x_size,
            y_size: d.y_size,
        }
    } else if let Ok(d) = dataset.extract::<PyRef<LocalDataset>>() {
        DatasetConfig {
            src: DatasetSrc::Local {
                samples_path: d.samples_path.clone(),
                labels_path: d.labels_path.clone(),
            },
            x_size: d.x_size,
            y_size: d.y_size,
        }
    } else {
        return Err(PyTypeError::new_err(
            "dataset must be an InlineDataset or LocalDataset",
        ));
    };

    let worker_count = worker_addrs.len();

    Ok(PyTrainingConfig {
        inner: TrainingConfig {
            worker_addrs,
            algorithm: AlgorithmConfig::AllReduce,
            serializer: serializer_cfg,
            dataset: dataset_config,
            optimizer: OptimizerConfig::GradientDescent { lr: optimizer.lr },
            loss_fn: loss_fn_cfg,
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
