use std::io::Write;
use std::num::NonZeroUsize;

use orchestrator::{
    TrainingEvent,
    configs::{
        ActFnConfig, AlgorithmConfig, DatasetConfig, DatasetSrc, LayerConfig, LossFnConfig,
        ModelConfig, OptimizerConfig, ParamGenConfig, StoreConfig, SynchronizerConfig,
        TrainingConfig,
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

    /// Saves the trained parameters to a CSV file.
    ///
    /// # Args
    /// * `path` - Output file path.
    /// * `output_sizes` - List of output sizes per layer, e.g. [8, 4, 1].
    /// * `input_size` - Input size of the first layer.
    ///
    /// # Errors
    /// Raises a `RuntimeError` if the file cannot be written.
    pub fn save(
        &self,
        path: &str,
        output_sizes: Vec<usize>,
        input_size: usize,
    ) -> PyResult<()> {
        let mut csv = String::from("layer,type,index,value\n");
        let mut offset = 0;
        let mut prev = input_size;

        for (layer_i, &out) in output_sizes.iter().enumerate() {
            let w_count = prev * out;
            let b_count = out;

            for i in 0..w_count {
                let v = self.params[offset + i];
                csv.push_str(&format!("{layer_i},weight,{i},{v}\n"));
            }
            offset += w_count;

            for i in 0..b_count {
                let v = self.params[offset + i];
                csv.push_str(&format!("{layer_i},bias,{i},{v}\n"));
            }
            offset += b_count;

            prev = out;
        }

        let mut file = std::fs::File::create(path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        file.write_all(csv.as_bytes())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(())
    }
}

/// A handle to an ongoing training session.
#[pyclass]
pub struct Session {
    inner: Option<orchestrator::Session>,
    max_epochs: usize,
    worker_count: usize,
}

#[pymethods]
impl Session {
    /// Blocks until training completes, showing a progress bar, and returns the trained model.
    ///
    /// # Errors
    /// Raises a `RuntimeError` if training fails.
    pub fn wait(&mut self, py: Python<'_>) -> PyResult<TrainedModel> {
        let session = self
            .inner
            .take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("session already consumed"))?;

        let max_epochs = self.max_epochs;
        let worker_count = self.worker_count;

        let params = py.allow_threads(|| {
            std::thread::spawn(move || {
                let mut rx = session.event_listener();
                let mut worker_epochs: Vec<usize> = vec![0; worker_count];
                let mut last_loss: Vec<Option<f32>> = vec![None; worker_count];
                let bar_width = 40usize;

                println!();
                println!();

                loop {
                    match rx.blocking_recv() {
                        Some(TrainingEvent::Loss { worker_id, losses }) => {
                            for loss in &losses {
                                if worker_id < worker_epochs.len() {
                                    worker_epochs[worker_id] += 1;
                                    last_loss[worker_id] = Some(*loss);
                                }
                            }

                            let current_epoch = *worker_epochs.iter().max().unwrap_or(&0);
                            let reported: Vec<f32> =
                                last_loss.iter().filter_map(|l| *l).collect();
                            let avg_loss =
                                reported.iter().sum::<f32>() / reported.len() as f32;

                            let filled =
                                ((current_epoch * bar_width) / max_epochs).min(bar_width);
                            print!(
                                "\x1b[2A\r  [{}{}] {}/{}\n  avg_loss={:.6}\n",
                                "█".repeat(filled),
                                "░".repeat(bar_width - filled),
                                current_epoch,
                                max_epochs,
                                avg_loss,
                            );
                            let _ = std::io::stdout().flush();
                        }
                        Some(TrainingEvent::Complete(params)) => {
                            let reported: Vec<f32> =
                                last_loss.iter().filter_map(|l| *l).collect();
                            let avg_loss = if reported.is_empty() {
                                0.0
                            } else {
                                reported.iter().sum::<f32>() / reported.len() as f32
                            };
                            print!(
                                "\x1b[2A\r  [{}] {}/{}\n  avg_loss={:.6}\n\n",
                                "█".repeat(bar_width),
                                max_epochs,
                                max_epochs,
                                avg_loss,
                            );
                            let _ = std::io::stdout().flush();
                            return Ok(params);
                        }
                        Some(TrainingEvent::Error(e)) => {
                            println!();
                            return Err(e.to_string());
                        }
                        Some(_) => continue,
                        None => return Err("session channel closed unexpectedly".into()),
                    }
                }
            })
            .join()
            .map_err(|_| "session thread panicked".to_string())?
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

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
    /// * `output_size` - Output size of this layer.
    /// * `sigmoid_amp` - If provided, applies a sigmoid activation with this amplitude.
    #[pyo3(signature = (output_size, sigmoid_amp=None))]
    pub fn dense(&mut self, output_size: usize, sigmoid_amp: Option<f32>) -> PyResult<()> {
        let output_size = NonZeroUsize::new(output_size).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("output_size must be greater than 0")
        })?;
        self.layers.push(LayerConfig::Dense {
            output_size,
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

    /// Sets barrier synchronization.
    pub fn barrier_sync(&mut self) -> PyResult<()> {
        self.synchronizer = SynchronizerConfig::Barrier;
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
    pub fn inline_dataset(
        &mut self,
        data: Vec<f32>,
        x_size: usize,
        y_size: usize,
    ) -> PyResult<()> {
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
        let x_size = NonZeroUsize::new(x_size).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("x_size must be greater than 0")
        })?;
        let y_size = NonZeroUsize::new(y_size).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("y_size must be greater than 0")
        })?;

        Ok(PyTrainingConfig {
            inner: TrainingConfig {
                worker_addrs: self.worker_addrs.clone(),
                algorithm: AlgorithmConfig::ParameterServer {
                    server_addrs: self.server_addrs.clone(),
                    synchronizer: self.synchronizer,
                    store: self.store,
                },
                dataset: DatasetConfig {
                    src: DatasetSrc::Inline { data },
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
            max_epochs: self.max_epochs,
            worker_count: self.worker_addrs.len(),
        })
    }
}

/// Opaque wrapper around a `TrainingConfig<String>`.
#[pyclass]
pub struct PyTrainingConfig {
    inner: TrainingConfig<String>,
    max_epochs: usize,
    worker_count: usize,
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
    /// # Errors
    /// Raises a `RuntimeError` if the session cannot be started.
    pub fn train(
        &self,
        py: Python<'_>,
        model: &PyModelConfig,
        training: &PyTrainingConfig,
    ) -> PyResult<Session> {
        let model = model.inner.clone();
        let max_epochs = training.max_epochs;
        let worker_count = training.worker_count;
        let training = training.inner.clone();

        let session = py.allow_threads(|| {
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