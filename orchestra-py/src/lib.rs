use std::num::NonZeroUsize;

use orchestrator::{
    configs::{
        ActFnConfig, AlgorithmConfig, DatasetConfig, LayerConfig, LossFnConfig, ModelConfig,
        OptimizerConfig, ParamGenConfig, StoreConfig, SynchronizerConfig, TrainingConfig,
    },
    train,
};
use pyo3::prelude::*;

/// The final trained model, holding the parameters received from all servers.
#[pyclass]
pub struct TrainedModel {
    params: Vec<f32>,
}

#[pymethods]
impl TrainedModel {
    /// Returns the trained model parameters as a Python list.
    pub fn weights(&self) -> Vec<f32> {
        self.params.clone()
    }
}

/// A handle to an ongoing training session.
#[pyclass]
pub struct Session {
    inner: Option<orchestrator::Session>,
}

#[pymethods]
impl Session {
    /// Blocks until training completes and returns the trained model.
    ///
    /// # Errors
    /// Raises a `RuntimeError` if training fails.
    pub fn wait(&mut self) -> PyResult<TrainedModel> {
        let session = self
            .inner
            .take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("session already consumed"))?;

        let params = session
            .wait()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(TrainedModel { params })
    }
}

/// Builder for the model configuration.
#[pyclass]
pub struct ModelBuilder {
    layers: Vec<LayerConfig>,
}

#[pymethods]
impl ModelBuilder {
    #[new]
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    /// Adds a dense layer.
    ///
    /// # Args
    /// * `n` - Input size.
    /// * `m` - Output size.
    /// * `sigmoid_amp` - If provided, applies a sigmoid activation with this amplitude.
    #[pyo3(signature = (n, m, sigmoid_amp=None))]
    pub fn dense(&mut self, n: usize, m: usize, sigmoid_amp: Option<f32>) -> PyResult<()> {
        self.layers.push(LayerConfig::Dense {
            dim: (n, m),
            init: ParamGenConfig::Kaiming,
            act_fn: sigmoid_amp.map(|amp| ActFnConfig::Sigmoid { amp }),
        });
        Ok(())
    }

    /// Builds and returns the model configuration.
    ///
    /// # Errors
    /// Raises a `ValueError` if no layers have been added.
    pub fn build(&self) -> PyResult<PyModelConfig> {
        if self.layers.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "model must have at least one layer",
            ));
        }
        Ok(PyModelConfig {
            inner: ModelConfig::Sequential {
                layers: self.layers.clone(),
            },
        })
    }
}

/// Opaque wrapper around a `ModelConfig`.
#[pyclass]
pub struct PyModelConfig {
    inner: ModelConfig,
}

/// Builder for the training configuration.
#[pyclass]
pub struct TrainingBuilder {
    worker_addrs: Vec<String>,
    server_addrs: Vec<String>,
    synchronizer: SynchronizerConfig,
    store: StoreConfig,
    lr: f32,
    batch_size: usize,
    max_epochs: usize,
    offline_epochs: usize,
    seed: Option<u64>,
    inline_data: Option<(Vec<f32>, usize, usize)>,
}

#[pymethods]
impl TrainingBuilder {
    #[new]
    pub fn new(worker_addrs: Vec<String>, server_addrs: Vec<String>) -> Self {
        Self {
            worker_addrs,
            server_addrs,
            synchronizer: SynchronizerConfig::NonBlocking,
            store: StoreConfig::Blocking,
            lr: 0.01,
            batch_size: 32,
            max_epochs: 100,
            offline_epochs: 0,
            seed: None,
            inline_data: None,
        }
    }

    /// Sets barrier synchronization with the given barrier size.
    pub fn barrier_sync(&mut self, barrier_size: usize) -> PyResult<()> {
        self.synchronizer = SynchronizerConfig::Barrier { barrier_size };
        Ok(())
    }

    /// Sets the learning rate.
    pub fn learning_rate(&mut self, lr: f32) -> PyResult<()> {
        self.lr = lr;
        Ok(())
    }

    /// Sets the batch size.
    pub fn batch_size(&mut self, batch_size: usize) -> PyResult<()> {
        self.batch_size = batch_size;
        Ok(())
    }

    /// Sets the maximum number of training epochs.
    pub fn max_epochs(&mut self, max_epochs: usize) -> PyResult<()> {
        self.max_epochs = max_epochs;
        Ok(())
    }

    /// Sets the random seed.
    pub fn seed(&mut self, seed: u64) -> PyResult<()> {
        self.seed = Some(seed);
        Ok(())
    }

    /// Sets the inline dataset from a flat list of floats.
    ///
    /// # Args
    /// * `data` - Flat list of floats, row-major.
    /// * `x_size` - Number of input features per sample.
    /// * `y_size` - Number of output features per sample.
    pub fn inline_dataset(&mut self, data: Vec<f32>, x_size: usize, y_size: usize) -> PyResult<()> {
        self.inline_data = Some((data, x_size, y_size));
        Ok(())
    }

    /// Builds and returns the training configuration.
    ///
    /// # Errors
    /// Raises a `ValueError` if required fields are missing or invalid.
    pub fn build(&self) -> PyResult<PyTrainingConfig> {
        let batch_size = NonZeroUsize::new(self.batch_size).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("batch_size must be greater than 0")
        })?;
        let max_epochs = NonZeroUsize::new(self.max_epochs).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("max_epochs must be greater than 0")
        })?;
        let (data, x_size, y_size) = self
            .inline_data
            .clone()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("dataset is required"))?;

        Ok(PyTrainingConfig {
            inner: TrainingConfig {
                worker_addrs: self.worker_addrs.clone(),
                algorithm: AlgorithmConfig::ParameterServer {
                    server_addrs: self.server_addrs.clone(),
                    synchronizer: self.synchronizer,
                    store: self.store,
                },
                dataset: DatasetConfig::Inline {
                    data,
                    x_size,
                    y_size,
                },
                optimizer: OptimizerConfig::GradientDescent { lr: self.lr },
                loss_fn: LossFnConfig::Mse,
                batch_size,
                max_epochs,
                offline_epochs: self.offline_epochs,
                seed: self.seed,
            },
        })
    }
}

/// Opaque wrapper around a `TrainingConfig<String>`.
#[pyclass]
pub struct PyTrainingConfig {
    inner: TrainingConfig<String>,
}

/// The main entry point for distributed training from Python.
#[pyclass]
pub struct Orchestrator;

#[pymethods]
impl Orchestrator {
    #[new]
    pub fn new() -> Self {
        Self
    }

    /// Starts a distributed training session.
    ///
    /// # Args
    /// * `model` - The model configuration produced by `ModelBuilder.build()`.
    /// * `training` - The training configuration produced by `TrainingBuilder.build()`.
    ///
    /// # Errors
    /// Raises a `RuntimeError` if the session cannot be started.
    pub fn train(&self, model: &PyModelConfig, training: &PyTrainingConfig) -> PyResult<Session> {
        let session = train(model.inner.clone(), training.inner.clone())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(Session {
            inner: Some(session),
        })
    }
}

#[pymodule]
fn orchestra(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Orchestrator>()?;
    m.add_class::<ModelBuilder>()?;
    m.add_class::<TrainingBuilder>()?;
    m.add_class::<PyModelConfig>()?;
    m.add_class::<PyTrainingConfig>()?;
    m.add_class::<Session>()?;
    m.add_class::<TrainedModel>()?;
    Ok(())
}
