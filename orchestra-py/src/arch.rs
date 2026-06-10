use std::num::NonZeroUsize;

use comms::floats::Float01;
use orchestrator::configs::{ActFnConfig, LayerConfig, ModelConfig, ParamGenConfig};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
};

use super::{
    activations::{ReLU, Sigmoid, Tanh},
    initialization::{
        Const, Kaiming, Lecun, LecunUniform, Normal, Uniform, UniformInclusive, Xavier,
        XavierUniform,
    },
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
    Tanh(f32),
    ReLU(Float01),
}

/// Converts a Python initializer object to a `PyInit`.
///
/// Returns a `TypeError` if the object is not a recognised initializer.
pub fn extract_init(obj: &Bound<'_, PyAny>) -> PyResult<PyInit> {
    if obj.is_instance_of::<Kaiming>() {
        Ok(PyInit::Kaiming)
    } else if obj.is_instance_of::<Xavier>() {
        Ok(PyInit::Xavier)
    } else if obj.is_instance_of::<Lecun>() {
        Ok(PyInit::Lecun)
    } else if obj.is_instance_of::<XavierUniform>() {
        Ok(PyInit::XavierUniform)
    } else if obj.is_instance_of::<LecunUniform>() {
        Ok(PyInit::LecunUniform)
    } else if let Ok(c) = obj.extract::<PyRef<Const>>() {
        Ok(PyInit::Const(c.value))
    } else if let Ok(u) = obj.extract::<PyRef<Uniform>>() {
        Ok(PyInit::Uniform(u.low, u.high))
    } else if let Ok(u) = obj.extract::<PyRef<UniformInclusive>>() {
        Ok(PyInit::UniformInclusive(u.low, u.high))
    } else if let Ok(n) = obj.extract::<PyRef<Normal>>() {
        Ok(PyInit::Normal(n.mean, n.std_dev))
    } else {
        Err(PyTypeError::new_err("init must be a parameter initializer"))
    }
}

/// Converts an optional Python activation function object to a `PyActFn`.
///
/// Returns a `TypeError` if the object is not a recognised activation function.
pub fn extract_act_fn(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Option<PyActFn>> {
    match obj {
        None => Ok(None),
        Some(a) => {
            if let Ok(s) = a.extract::<PyRef<Sigmoid>>() {
                Ok(Some(PyActFn::Sigmoid(s.amp)))
            } else if let Ok(t) = a.extract::<PyRef<Tanh>>() {
                Ok(Some(PyActFn::Tanh(t.amp)))
            } else if let Ok(r) = a.extract::<PyRef<ReLU>>() {
                let Some(slope) = Float01::new(r.slope) else {
                    return Err(PyTypeError::new_err(
                        "ReLu's slope must be a float between 0 and 1",
                    ));
                };

                Ok(Some(PyActFn::ReLU(slope)))
            } else {
                Err(PyTypeError::new_err(
                    "act_fn must be an activation function or None",
                ))
            }
        }
    }
}

/// A fully-connected dense layer.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct Dense {
    pub output_size: NonZeroUsize,
    pub init: PyInit,
    pub act_fn: Option<PyActFn>,
}

#[pymethods]
impl Dense {
    /// Creates a dense layer configuration.
    ///
    /// # Args
    /// * `output_size` - Number of output neurons.
    /// * `init` - Parameter initializer (e.g. `Kaiming()`, `Const(0.0)`).
    /// * `act_fn` - Optional activation function (e.g. `Sigmoid()`). Defaults to `None`.
    ///
    /// # Returns
    /// A dense layer configuration.
    ///
    /// # Errors
    /// Raises a `TypeError` if `init` is not a supported initializer.
    /// Raises a `TypeError` if `act_fn` is not a supported activation function.
    /// Raises a `ValueError` if `output_size` is zero.
    #[new]
    #[pyo3(signature = (output_size, init, act_fn = None))]
    pub fn new(
        output_size: usize,
        init: &Bound<'_, PyAny>,
        act_fn: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let output_size = NonZeroUsize::new(output_size)
            .ok_or_else(|| PyValueError::new_err("output_size must be greater than 0"))?;

        Ok(Self {
            output_size,
            init: extract_init(init)?,
            act_fn: extract_act_fn(act_fn)?,
        })
    }
}

impl Dense {
    pub fn to_layer_config(&self) -> LayerConfig {
        LayerConfig::Dense {
            output_size: self.output_size,
            init: py_init_to_config(&self.init),
            act_fn: self.act_fn.as_ref().map(py_act_fn_to_config),
        }
    }
}

/// A 2D convolutional layer.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct Conv2d {
    pub input_dim: (NonZeroUsize, NonZeroUsize, NonZeroUsize),
    pub kernel_dim: (NonZeroUsize, NonZeroUsize, NonZeroUsize),
    pub stride: NonZeroUsize,
    pub padding: usize,
    pub init: PyInit,
    pub act_fn: Option<PyActFn>,
}

#[pymethods]
impl Conv2d {
    /// Creates a Conv2d layer configuration.
    ///
    /// # Args
    /// * `input_dim` - Tuple `(in_channels, height, width)` of the input tensor.
    /// * `kernel_dim` - Tuple `(filters, in_channels, kernel_size)`. The kernel is square.
    /// * `stride` - Convolution stride. Must be > 0.
    /// * `padding` - Zero-padding applied to each spatial side of the input.
    /// * `init` - Parameter initializer (e.g. `Kaiming()`, `Const(0.0)`).
    /// * `act_fn` - Optional activation function (e.g. `Sigmoid()`). Defaults to `None`.
    ///
    /// # Returns
    /// A Conv2d layer configuration.
    ///
    /// # Errors
    /// Raises a `ValueError` if any dimension or stride is zero.
    /// Raises a `TypeError` if `init` or `act_fn` are not supported types.
    #[new]
    #[pyo3(signature = (input_dim, kernel_dim, stride, padding, init, act_fn = None))]
    pub fn new(
        input_dim: (usize, usize, usize),
        kernel_dim: (usize, usize, usize),
        stride: usize,
        padding: usize,
        init: &Bound<'_, PyAny>,
        act_fn: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let make_nonzero = |v: usize, name: &'static str| {
            NonZeroUsize::new(v).ok_or_else(|| PyValueError::new_err(format!("{name} must be > 0")))
        };

        Ok(Self {
            input_dim: (
                make_nonzero(input_dim.0, "input_dim.channels")?,
                make_nonzero(input_dim.1, "input_dim.height")?,
                make_nonzero(input_dim.2, "input_dim.width")?,
            ),
            kernel_dim: (
                make_nonzero(kernel_dim.0, "kernel_dim.filters")?,
                make_nonzero(kernel_dim.1, "kernel_dim.in_channels")?,
                make_nonzero(kernel_dim.2, "kernel_dim.kernel_size")?,
            ),
            stride: make_nonzero(stride, "stride")?,
            padding,
            init: extract_init(init)?,
            act_fn: extract_act_fn(act_fn)?,
        })
    }
}

impl Conv2d {
    pub fn to_layer_config(&self) -> LayerConfig {
        LayerConfig::Conv {
            input_dim: self.input_dim,
            kernel_dim: self.kernel_dim,
            stride: self.stride,
            padding: self.padding,
            init: py_init_to_config(&self.init),
            act_fn: self.act_fn.as_ref().map(py_act_fn_to_config),
        }
    }
}

/// A sequential model — layers are applied in order.
#[pyclass]
pub struct Sequential {
    pub inner: ModelConfig,
}

#[pymethods]
impl Sequential {
    /// Creates a sequential model configuration.
    ///
    /// # Args
    /// * `layers` - List of `Dense` or `Conv2d` layers. At least one required.
    ///
    /// # Returns
    /// A sequential model configuration.
    ///
    /// # Errors
    /// Raises a `ValueError` if `layers` is empty.
    /// Raises a `TypeError` if any element is not a `Dense` or `Conv2d` instance.
    #[new]
    pub fn new(layers: Vec<Bound<'_, PyAny>>) -> PyResult<Self> {
        if layers.is_empty() {
            return Err(PyValueError::new_err("model must have at least one layer"));
        }

        let layer_configs: PyResult<Vec<_>> = layers
            .iter()
            .map(|l| {
                if let Ok(d) = l.extract::<PyRef<Dense>>() {
                    Ok(d.to_layer_config())
                } else if let Ok(c) = l.extract::<PyRef<Conv2d>>() {
                    Ok(c.to_layer_config())
                } else {
                    Err(PyTypeError::new_err(
                        "each layer must be a Dense or Conv2d instance",
                    ))
                }
            })
            .collect();

        Ok(Self {
            inner: ModelConfig {
                layers: layer_configs?,
            },
        })
    }
}

fn py_init_to_config(init: &PyInit) -> ParamGenConfig {
    match init {
        PyInit::Kaiming => ParamGenConfig::Kaiming,
        PyInit::Xavier => ParamGenConfig::Xavier,
        PyInit::Lecun => ParamGenConfig::Lecun,
        PyInit::XavierUniform => ParamGenConfig::XavierUniform,
        PyInit::LecunUniform => ParamGenConfig::LecunUniform,
        PyInit::Const(v) => ParamGenConfig::Const { value: *v },
        PyInit::Uniform(low, high) => ParamGenConfig::Uniform {
            low: *low,
            high: *high,
        },
        PyInit::UniformInclusive(low, high) => ParamGenConfig::UniformInclusive {
            low: *low,
            high: *high,
        },
        PyInit::Normal(mean, std_dev) => ParamGenConfig::Normal {
            mean: *mean,
            std_dev: *std_dev,
        },
    }
}

fn py_act_fn_to_config(act_fn: &PyActFn) -> ActFnConfig {
    match act_fn {
        PyActFn::Sigmoid(amp) => ActFnConfig::Sigmoid { amp: *amp },
        PyActFn::Tanh(amp) => ActFnConfig::Tanh { amp: *amp },
        PyActFn::ReLU(slope) => ActFnConfig::ReLU { slope: *slope },
    }
}
