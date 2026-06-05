use std::num::NonZeroUsize;

use comms::floats::{FloatNonNegative, FloatPositive};
use orchestrator::configs::{
    DataSrc, DatasetConfig, EarlyStoppingConfig, LossFnConfig, OptimizerConfig, SerializerConfig,
    StoreConfig, SynchronizerConfig,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::datasets::{InlineDataset, LocalDataset};
use crate::loss_fns::{CrossEntropy, Mse};
use crate::optimizers::GradientDescent;
use crate::serializer::{BaseSerializer, SparseSerializer};
use crate::store::{BlockingStore, WildStore};
use crate::sync::{BarrierSync, NonBlockingSync};

/// Converts a `usize` to `NonZeroUsize`, returning a `ValueError` if it is zero.
pub fn parse_nonzero(n: usize, name: &'static str) -> PyResult<NonZeroUsize> {
    NonZeroUsize::new(n)
        .ok_or_else(|| PyValueError::new_err(format!("{name} must be greater than 0")))
}

/// Converts an optional early-stopping tolerance to an `EarlyStoppingConfig`.
///
/// Returns a `ValueError` if the tolerance is zero or negative.
pub fn extract_early_stopping(
    tolerance: Option<f64>,
) -> PyResult<Option<EarlyStoppingConfig>> {
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

/// Converts a Python optimizer object to an `OptimizerConfig`.
///
/// Returns a `TypeError` if the object is not a recognised optimizer, or a
/// `ValueError` if the learning rate is not positive.
pub fn extract_optimizer(obj: &Bound<'_, PyAny>) -> PyResult<OptimizerConfig> {
    if let Ok(gd) = obj.extract::<PyRef<GradientDescent>>() {
        if gd.lr <= 0.0 {
            return Err(PyValueError::new_err(
                "learning rate must be a positive number",
            ));
        }
        Ok(OptimizerConfig::GradientDescent {
            lr: FloatPositive::new(gd.lr).unwrap(),
        })
    } else {
        Err(PyTypeError::new_err("optimizer must be GradientDescent(lr=...)"))
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
        Err(PyTypeError::new_err("loss_fn must be Mse() or CrossEntropy()"))
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
        Err(PyTypeError::new_err("store must be BlockingStore() or WildStore()"))
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
