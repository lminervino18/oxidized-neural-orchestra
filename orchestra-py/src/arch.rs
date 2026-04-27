use std::num::NonZeroUsize;

use orchestrator::configs::{ActFnConfig, LayerConfig, ModelConfig, ParamGenConfig};
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

        Ok(Self {
            output_size,
            init,
            act_fn,
        })
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

        LayerConfig::Dense {
            output_size: self.output_size,
            init,
            act_fn,
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
            NonZeroUsize::new(v)
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("{name} must be > 0")))
        };

        let input_dim = (
            make_nonzero(input_dim.0, "input_dim.channels")?,
            make_nonzero(input_dim.1, "input_dim.height")?,
            make_nonzero(input_dim.2, "input_dim.width")?,
        );
        let kernel_dim = (
            make_nonzero(kernel_dim.0, "kernel_dim.filters")?,
            make_nonzero(kernel_dim.1, "kernel_dim.in_channels")?,
            make_nonzero(kernel_dim.2, "kernel_dim.kernel_size")?,
        );
        let stride = make_nonzero(stride, "stride")?;

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

        Ok(Self {
            input_dim,
            kernel_dim,
            stride,
            padding,
            init,
            act_fn,
        })
    }
}

impl Conv2d {
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

        LayerConfig::Conv {
            input_dim: self.input_dim,
            kernel_dim: self.kernel_dim,
            stride: self.stride,
            padding: self.padding,
            init,
            act_fn,
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
            return Err(pyo3::exceptions::PyValueError::new_err(
                "model must have at least one layer",
            ));
        }

        let layer_configs: PyResult<Vec<_>> = layers
            .iter()
            .map(|l| {
                if let Ok(d) = l.extract::<PyRef<Dense>>() {
                    Ok(d.to_layer_config())
                } else if let Ok(c) = l.extract::<PyRef<Conv2d>>() {
                    Ok(c.to_layer_config())
                } else {
                    Err(pyo3::exceptions::PyTypeError::new_err(
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
