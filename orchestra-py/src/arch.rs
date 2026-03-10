use std::num::NonZeroUsize;

use orchestrator::configs::{
    ActFnConfig, LayerConfig, ModelConfig, ParamGenConfig,
};
use pyo3::prelude::*;

use crate::activations::Sigmoid;
use crate::initialization::{
    Const, Kaiming, Lecun, LecunUniform, Normal, Uniform, UniformInclusive, Xavier, XavierUniform,
};

#[derive(Clone)]
pub enum PyInit {
    Kaiming,
    Xavier,
    Lecun,
    XavierUniform,
    LecunUniform,
    Const(f32),
    Uniform(f32, f32),
    UniformInclusive(f32, f32),
    Normal(f32, f32),
}

#[derive(Clone)]
pub enum PyActFn {
    Sigmoid(f32),
}

/// A fully-connected dense layer.
///
/// # Args
/// * `output_size` - Number of output neurons.
/// * `init` - Parameter initializer (e.g. `Kaiming()`, `Const(0.0)`).
/// * `act_fn` - Optional activation function (e.g. `Sigmoid()`). Defaults to `None`.
#[pyclass]
#[derive(Clone)]
pub struct Dense {
    pub output_size: NonZeroUsize,
    pub init: PyInit,
    pub act_fn: Option<PyActFn>,
}

#[pymethods]
impl Dense {
    #[new]
    #[pyo3(signature = (output_size, init, act_fn = None))]
    pub fn new(
        output_size: usize,
        init: &Bound<'_, PyAny>,
        act_fn: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let output_size = NonZeroUsize::new(output_size).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("output_size must be greater than 0")
        })?;

        let init = if init.is_instance_of::<Kaiming>() {
            PyInit::Kaiming
        } else if init.is_instance_of::<Xavier>() {
            PyInit::Xavier
        } else if init.is_instance_of::<Lecun>() {
            PyInit::Lecun
        } else if init.is_instance_of::<XavierUniform>() {
            PyInit::XavierUniform
        } else if init.is_instance_of::<LecunUniform>() {
            PyInit::LecunUniform
        } else if let Ok(c) = init.extract::<PyRef<Const>>() {
            PyInit::Const(c.value)
        } else if let Ok(u) = init.extract::<PyRef<Uniform>>() {
            PyInit::Uniform(u.low, u.high)
        } else if let Ok(u) = init.extract::<PyRef<UniformInclusive>>() {
            PyInit::UniformInclusive(u.low, u.high)
        } else if let Ok(n) = init.extract::<PyRef<Normal>>() {
            PyInit::Normal(n.mean, n.std_dev)
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "init must be a parameter initializer",
            ));
        };

        let act_fn = match act_fn {
            None => None,
            Some(a) => {
                if let Ok(s) = a.extract::<PyRef<Sigmoid>>() {
                    Some(PyActFn::Sigmoid(s.amp))
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "act_fn must be an activation function or None",
                    ));
                }
            }
        };

        Ok(Self { output_size, init, act_fn })
    }
}

impl Dense {
    pub fn to_layer_config(&self) -> LayerConfig {
        let init = match self.init {
            PyInit::Kaiming => ParamGenConfig::Kaiming,
            PyInit::Xavier => ParamGenConfig::Xavier,
            PyInit::Lecun => ParamGenConfig::Lecun,
            PyInit::XavierUniform => ParamGenConfig::XavierUniform,
            PyInit::LecunUniform => ParamGenConfig::LecunUniform,
            PyInit::Const(v) => ParamGenConfig::Const { value: v },
            PyInit::Uniform(low, high) => ParamGenConfig::Uniform { low, high },
            PyInit::UniformInclusive(low, high) => ParamGenConfig::UniformInclusive { low, high },
            PyInit::Normal(mean, std_dev) => ParamGenConfig::Normal { mean, std_dev },
        };
        let act_fn = self.act_fn.as_ref().map(|a| match a {
            PyActFn::Sigmoid(amp) => ActFnConfig::Sigmoid { amp: *amp },
        });
        LayerConfig::Dense { output_size: self.output_size, init, act_fn }
    }
}

/// A sequential model — layers are applied in order.
///
/// # Args
/// * `layers` - List of `Dense` layers.
#[pyclass]
pub struct Sequential {
    pub inner: ModelConfig,
}

#[pymethods]
impl Sequential {
    #[new]
    pub fn new(layers: Vec<PyRef<Dense>>) -> PyResult<Self> {
        if layers.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "model must have at least one layer",
            ));
        }
        let layer_configs = layers.iter().map(|l| l.to_layer_config()).collect();
        Ok(Self { inner: ModelConfig { layers: layer_configs } })
    }
}