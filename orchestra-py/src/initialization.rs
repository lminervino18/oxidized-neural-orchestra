use pyo3::prelude::*;

/// Kaiming (He) initialization.
///
/// # Args
/// This constructor does not take arguments.
///
/// # Returns
/// A Kaiming initializer configuration.
///
/// # Errors
/// This constructor does not return errors.
///
/// # Panics
/// This constructor does not panic.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct Kaiming;

#[pymethods]
impl Kaiming {
    #[new]
    pub fn new() -> Self {
        Self
    }
}

/// Xavier (Glorot) initialization.
///
/// # Args
/// This constructor does not take arguments.
///
/// # Returns
/// A Xavier initializer configuration.
///
/// # Errors
/// This constructor does not return errors.
///
/// # Panics
/// This constructor does not panic.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct Xavier;

#[pymethods]
impl Xavier {
    #[new]
    pub fn new() -> Self {
        Self
    }
}

/// Lecun initialization.
///
/// # Args
/// This constructor does not take arguments.
///
/// # Returns
/// A Lecun initializer configuration.
///
/// # Errors
/// This constructor does not return errors.
///
/// # Panics
/// This constructor does not panic.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct Lecun;

#[pymethods]
impl Lecun {
    #[new]
    pub fn new() -> Self {
        Self
    }
}

/// Xavier uniform initialization.
///
/// # Args
/// This constructor does not take arguments.
///
/// # Returns
/// A Xavier uniform initializer configuration.
///
/// # Errors
/// This constructor does not return errors.
///
/// # Panics
/// This constructor does not panic.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct XavierUniform;

#[pymethods]
impl XavierUniform {
    #[new]
    pub fn new() -> Self {
        Self
    }
}

/// Lecun uniform initialization.
///
/// # Args
/// This constructor does not take arguments.
///
/// # Returns
/// A Lecun uniform initializer configuration.
///
/// # Errors
/// This constructor does not return errors.
///
/// # Panics
/// This constructor does not panic.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct LecunUniform;

#[pymethods]
impl LecunUniform {
    #[new]
    pub fn new() -> Self {
        Self
    }
}

/// Constant initialization.
///
/// # Args
/// * `value` - The constant value to initialize all parameters with.
///
/// # Returns
/// A constant initializer configuration.
///
/// # Errors
/// This constructor does not return errors.
///
/// # Panics
/// This constructor does not panic.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct Const {
    pub value: f32,
}

#[pymethods]
impl Const {
    #[new]
    pub fn new(value: f32) -> Self {
        Self { value }
    }
}

/// Uniform random initialization.
///
/// # Args
/// * `low` - Lower bound of the uniform distribution.
/// * `high` - Upper bound of the uniform distribution.
///
/// # Returns
/// A uniform initializer configuration.
///
/// # Errors
/// This constructor does not return errors.
///
/// # Panics
/// This constructor does not panic.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct Uniform {
    pub low: f32,
    pub high: f32,
}

#[pymethods]
impl Uniform {
    #[new]
    pub fn new(low: f32, high: f32) -> Self {
        Self { low, high }
    }
}

/// Uniform inclusive random initialization.
///
/// # Args
/// * `low` - Lower bound of the uniform distribution (inclusive).
/// * `high` - Upper bound of the uniform distribution (inclusive).
///
/// # Returns
/// A uniform inclusive initializer configuration.
///
/// # Errors
/// This constructor does not return errors.
///
/// # Panics
/// This constructor does not panic.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct UniformInclusive {
    pub low: f32,
    pub high: f32,
}

#[pymethods]
impl UniformInclusive {
    #[new]
    pub fn new(low: f32, high: f32) -> Self {
        Self { low, high }
    }
}

/// Normal (Gaussian) random initialization.
///
/// # Args
/// * `mean` - Mean of the normal distribution.
/// * `std_dev` - Standard deviation of the normal distribution.
///
/// # Returns
/// A normal initializer configuration.
///
/// # Errors
/// This constructor does not return errors.
///
/// # Panics
/// This constructor does not panic.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct Normal {
    pub mean: f32,
    pub std_dev: f32,
}

#[pymethods]
impl Normal {
    #[new]
    pub fn new(mean: f32, std_dev: f32) -> Self {
        Self { mean, std_dev }
    }
}
