use std::num::NonZeroUsize;

use comms::floats::{Float01, FloatNonNegative, FloatPositive};
use orchestrator::configs::{
    DataSrc, DatasetConfig, EarlyStoppingConfig, LossFnConfig, OptimizerConfig, SerializerConfig,
    StoreConfig, SynchronizerConfig,
};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
};

use super::{
    datasets::{InlineDataset, LocalDataset},
    loss_fns::{CrossEntropy, Mse},
    optimizers::{Adam, GradientDescent, GradientDescentWithMomentum},
    serializer::{BaseSerializer, SparseSerializer},
    store::{BlockingStore, WildStore},
    sync::{BarrierSync, NonBlockingSync},
};

/// Converts a `usize` to `NonZeroUsize`, returning a `ValueError` if it is zero.
pub fn parse_nonzero(n: usize, name: &'static str) -> PyResult<NonZeroUsize> {
    NonZeroUsize::new(n)
        .ok_or_else(|| PyValueError::new_err(format!("{name} must be greater than 0")))
}

/// Converts an optional early-stopping tolerance to an `EarlyStoppingConfig`.
///
/// Returns a `ValueError` if the tolerance is zero or negative.
pub fn extract_early_stopping(tolerance: Option<f64>) -> PyResult<Option<EarlyStoppingConfig>> {
    match tolerance {
        Some(t) if t.is_sign_negative() || t == 0.0 => Err(PyValueError::new_err(
            "early stopping tolerance must be a non negative number",
        )),
        Some(t) => Ok(Some(EarlyStoppingConfig {
            tolerance: FloatNonNegative::new(t).unwrap(),
        })),
        None => Ok(None),
    }
}

/// Validates that a value is strictly positive, returning a `ValueError` otherwise.
fn extract_positive(value: f32, name: &str) -> PyResult<FloatPositive> {
    FloatPositive::new(value)
        .ok_or_else(|| PyValueError::new_err(format!("{name} must be a positive number")))
}

/// Validates that a value is between `0.0` and `1.0`, returning a `ValueError` otherwise.
fn extract_unit(value: f32, name: &str) -> PyResult<Float01> {
    Float01::new(value)
        .ok_or_else(|| PyValueError::new_err(format!("{name} must be between 0.0 and 1.0")))
}

/// Converts a Python optimizer object to an `OptimizerConfig`.
///
/// Returns a `TypeError` if the object is not a recognised optimizer, or a
/// `ValueError` if any hyperparameter is out of range.
pub fn extract_optimizer(obj: &Bound<'_, PyAny>) -> PyResult<OptimizerConfig> {
    if let Ok(gd) = obj.extract::<PyRef<GradientDescent>>() {
        Ok(OptimizerConfig::GradientDescent {
            lr: extract_positive(gd.lr, "learning rate")?,
        })
    } else if let Ok(gdm) = obj.extract::<PyRef<GradientDescentWithMomentum>>() {
        Ok(OptimizerConfig::GradientDescentWithMomentum {
            lr: extract_positive(gdm.lr, "learning rate")?,
            mu: extract_unit(gdm.mu, "momentum")?,
        })
    } else if let Ok(adam) = obj.extract::<PyRef<Adam>>() {
        Ok(OptimizerConfig::Adam {
            lr: extract_positive(adam.lr, "learning rate")?,
            b1: extract_unit(adam.b1, "b1")?,
            b2: extract_unit(adam.b2, "b2")?,
            eps: extract_positive(adam.eps, "eps")?,
        })
    } else {
        Err(PyTypeError::new_err(
            "optimizer must be GradientDescent(lr=...), \
             GradientDescentWithMomentum(lr=..., mu=...) or Adam(lr=..., b1=..., b2=..., eps=...)",
        ))
    }
}

/// Converts a Python loss function object to a `LossFnConfig`.
///
/// Returns a `TypeError` if the object is not a recognised loss function.
pub fn extract_loss_fn(obj: &Bound<'_, PyAny>) -> PyResult<LossFnConfig> {
    if obj.is_instance_of::<Mse>() {
        Ok(LossFnConfig::Mse)
    } else if obj.is_instance_of::<CrossEntropy>() {
        Ok(LossFnConfig::CrossEntropy)
    } else {
        Err(PyTypeError::new_err(
            "loss_fn must be Mse() or CrossEntropy()",
        ))
    }
}

/// Converts an optional Python serializer object to a `SerializerConfig`.
///
/// `None` or `BaseSerializer()` both map to `SerializerConfig::Base`.
/// Returns a `TypeError` if the object is not a recognised serializer.
pub fn extract_serializer(obj: Option<&Bound<'_, PyAny>>) -> PyResult<SerializerConfig> {
    match obj {
        None => Ok(SerializerConfig::Base),
        Some(s) if s.is_instance_of::<BaseSerializer>() => Ok(SerializerConfig::Base),
        Some(s) => {
            if let Ok(sparse) = s.extract::<PyRef<SparseSerializer>>() {
                Ok(SerializerConfig::SparseCapable { r: sparse.r })
            } else {
                Err(PyTypeError::new_err(
                    "serializer must be BaseSerializer() or SparseSerializer(r=...)",
                ))
            }
        }
    }
}

/// Converts a Python sync object to a `SynchronizerConfig`.
///
/// Returns a `TypeError` if the object is not a recognised synchronizer.
pub fn extract_synchronizer(obj: &Bound<'_, PyAny>) -> PyResult<SynchronizerConfig> {
    if obj.is_instance_of::<BarrierSync>() {
        Ok(SynchronizerConfig::Barrier)
    } else if obj.is_instance_of::<NonBlockingSync>() {
        Ok(SynchronizerConfig::NonBlocking)
    } else {
        Err(PyTypeError::new_err(
            "sync must be BarrierSync() or NonBlockingSync()",
        ))
    }
}

/// Converts a Python store object to a `StoreConfig`.
///
/// Returns a `TypeError` if the object is not a recognised store.
pub fn extract_store(obj: &Bound<'_, PyAny>) -> PyResult<StoreConfig> {
    if obj.is_instance_of::<BlockingStore>() {
        Ok(StoreConfig::Blocking)
    } else if obj.is_instance_of::<WildStore>() {
        Ok(StoreConfig::Wild)
    } else {
        Err(PyTypeError::new_err(
            "store must be BlockingStore() or WildStore()",
        ))
    }
}

/// Converts a Python dataset object to a `DatasetConfig`.
///
/// Returns a `TypeError` if the object is not an `InlineDataset` or `LocalDataset`.
pub fn extract_dataset(obj: &Bound<'_, PyAny>) -> PyResult<DatasetConfig> {
    if let Ok(d) = obj.extract::<PyRef<InlineDataset>>() {
        Ok(DatasetConfig {
            src: DataSrc::Inline {
                samples: d.samples.clone(),
                labels: d.labels.clone(),
            },
            x_size: d.x_size,
            y_size: d.y_size,
        })
    } else if let Ok(d) = obj.extract::<PyRef<LocalDataset>>() {
        Ok(DatasetConfig {
            src: DataSrc::Local {
                samples_path: d.samples_path.clone(),
                labels_path: d.labels_path.clone(),
            },
            x_size: d.x_size,
            y_size: d.y_size,
        })
    } else {
        Err(PyTypeError::new_err(
            "dataset must be an InlineDataset or LocalDataset",
        ))
    }
}
