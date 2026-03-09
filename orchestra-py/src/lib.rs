use std::io::Write;
use std::num::NonZeroUsize;

use orchestrator::{
    configs::{
        ActFnConfig, AlgorithmConfig, DatasetConfig, DatasetSrc, LayerConfig, LossFnConfig,
        ModelConfig, OptimizerConfig, ParamGenConfig, StoreConfig, SynchronizerConfig,
        TrainingConfig,
    },
    train, TrainingEvent,
};
use pyo3::prelude::*;

const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

// ---------------------------------------------------------------------------
// Activation functions
// ---------------------------------------------------------------------------

#[pyclass]
#[derive(Clone)]
pub struct Sigmoid {
    amp: f32,
}

#[pymethods]
impl Sigmoid {
    #[new]
    #[pyo3(signature = (amp = 1.0))]
    pub fn new(amp: f32) -> Self {
        Self { amp }
    }
}

// ---------------------------------------------------------------------------
// Parameter initializers
// ---------------------------------------------------------------------------

#[pyclass]
#[derive(Clone)]
pub struct Kaiming;

#[pymethods]
impl Kaiming {
    #[new]
    pub fn new() -> Self {
        Self
    }
}

#[pyclass]
#[derive(Clone)]
pub struct Const {
    value: f32,
}

#[pymethods]
impl Const {
    #[new]
    pub fn new(value: f32) -> Self {
        Self { value }
    }
}

// ---------------------------------------------------------------------------
// Layers
// ---------------------------------------------------------------------------

#[derive(Clone)]
enum PyInit {
    Kaiming,
    Const(f32),
}

#[derive(Clone)]
enum PyActFn {
    Sigmoid(f32),
}

#[pyclass]
#[derive(Clone)]
pub struct Dense {
    output_size: NonZeroUsize,
    init: PyInit,
    act_fn: Option<PyActFn>,
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
        } else if let Ok(c) = init.extract::<PyRef<Const>>() {
            PyInit::Const(c.value)
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "init must be Kaiming() or Const(value)",
            ));
        };

        let act_fn = match act_fn {
            None => None,
            Some(a) => {
                if let Ok(s) = a.extract::<PyRef<Sigmoid>>() {
                    Some(PyActFn::Sigmoid(s.amp))
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "act_fn must be Sigmoid(amp) or None",
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
    fn to_layer_config(&self) -> LayerConfig {
        let init = match self.init {
            PyInit::Kaiming => ParamGenConfig::Kaiming,
            PyInit::Const(v) => ParamGenConfig::Const { value: v },
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

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

#[pyclass]
pub struct Sequential {
    inner: ModelConfig,
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
        Ok(Self {
            inner: ModelConfig {
                layers: layer_configs,
            },
        })
    }
}

// ---------------------------------------------------------------------------
// Datasets
// ---------------------------------------------------------------------------

#[pyclass]
#[derive(Clone)]
pub struct InlineDataset {
    data: Vec<f32>,
    x_size: NonZeroUsize,
    y_size: NonZeroUsize,
}

#[pymethods]
impl InlineDataset {
    #[new]
    pub fn new(data: Vec<f32>, x_size: usize, y_size: usize) -> PyResult<Self> {
        let x_size = NonZeroUsize::new(x_size).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("x_size must be greater than 0")
        })?;
        let y_size = NonZeroUsize::new(y_size).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("y_size must be greater than 0")
        })?;
        Ok(Self {
            data,
            x_size,
            y_size,
        })
    }
}

// ---------------------------------------------------------------------------
// Optimizers
// ---------------------------------------------------------------------------

#[pyclass]
#[derive(Clone)]
pub struct GradientDescent {
    lr: f32,
}

#[pymethods]
impl GradientDescent {
    #[new]
    pub fn new(lr: f32) -> Self {
        Self { lr }
    }
}

// ---------------------------------------------------------------------------
// Synchronizers
// ---------------------------------------------------------------------------

#[pyclass]
#[derive(Clone)]
pub struct BarrierSync;

#[pymethods]
impl BarrierSync {
    #[new]
    pub fn new() -> Self {
        Self
    }
}

#[pyclass]
#[derive(Clone)]
pub struct NonBlockingSync;

#[pymethods]
impl NonBlockingSync {
    #[new]
    pub fn new() -> Self {
        Self
    }
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

#[pyclass]
#[derive(Clone)]
pub struct BlockingStore;

#[pymethods]
impl BlockingStore {
    #[new]
    pub fn new() -> Self {
        Self
    }
}

#[pyclass]
#[derive(Clone)]
pub struct WildStore;

#[pymethods]
impl WildStore {
    #[new]
    pub fn new() -> Self {
        Self
    }
}

// ---------------------------------------------------------------------------
// Training config
// ---------------------------------------------------------------------------

#[pyclass]
pub struct PyTrainingConfig {
    inner: TrainingConfig<String>,
    max_epochs: usize,
    worker_count: usize,
}

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

#[pyclass]
pub struct Session {
    inner: Option<orchestrator::Session>,
    max_epochs: usize,
    worker_count: usize,
}

#[pymethods]
impl Session {
    pub fn wait(&mut self, py: Python<'_>) -> PyResult<TrainedModel> {
        let session = self
            .inner
            .take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("session already consumed"))?;

        let max_epochs = self.max_epochs;
        let worker_count = self.worker_count;

        let params = py
            .allow_threads(|| {
                std::thread::spawn(move || {
                    let mut rx = session.event_listener();
                    let mut worker_epochs: Vec<usize> = vec![0; worker_count];
                    let mut last_loss: Vec<Option<f32>> = vec![None; worker_count];
                    let mut spinner_i = 0usize;
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
                                let avg_loss = reported.iter().sum::<f32>() / reported.len() as f32;
                                let filled =
                                    ((current_epoch * bar_width) / max_epochs).min(bar_width);
                                let spinner = SPINNER[spinner_i % SPINNER.len()];
                                spinner_i += 1;
                                print!(
                                    "\x1b[2A\r  {} [{}{}] {}/{}\n  avg_loss={:.6}\n",
                                    spinner,
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
                                    "\x1b[2A\r  ✓ [{}] {}/{}\n  avg_loss={:.6}\n\n",
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

// ---------------------------------------------------------------------------
// TrainedModel
// ---------------------------------------------------------------------------

#[pyclass]
pub struct TrainedModel {
    params: Vec<f32>,
}

#[pymethods]
impl TrainedModel {
    pub fn weights(&self) -> Vec<f32> {
        self.params.clone()
    }

    pub fn save(&self, path: &str, output_sizes: Vec<usize>, input_size: usize) -> PyResult<()> {
        let mut csv = String::from("layer,type,index,value\n");
        let mut offset = 0;
        let mut prev = input_size;

        for (layer_i, &out) in output_sizes.iter().enumerate() {
            let w_count = prev * out;
            let b_count = out;
            for i in 0..w_count {
                csv.push_str(&format!(
                    "{layer_i},weight,{i},{}\n",
                    self.params[offset + i]
                ));
            }
            offset += w_count;
            for i in 0..b_count {
                csv.push_str(&format!("{layer_i},bias,{i},{}\n", self.params[offset + i]));
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

// ---------------------------------------------------------------------------
// Functions
// ---------------------------------------------------------------------------

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
    dataset: PyRef<InlineDataset>,
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

    let worker_count = worker_addrs.len();

    Ok(PyTrainingConfig {
        inner: TrainingConfig {
            worker_addrs,
            algorithm: AlgorithmConfig::ParameterServer {
                server_addrs,
                synchronizer,
                store: store_cfg,
            },
            dataset: DatasetConfig {
                src: DatasetSrc::Inline {
                    data: dataset.data.clone(),
                },
                x_size: dataset.x_size,
                y_size: dataset.y_size,
            },
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

#[pymodule]
fn _orchestra(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Sequential>()?;
    m.add_class::<Session>()?;
    m.add_class::<TrainedModel>()?;
    m.add_class::<PyTrainingConfig>()?;
    m.add_class::<Dense>()?;
    m.add_class::<Sigmoid>()?;
    m.add_class::<Kaiming>()?;
    m.add_class::<Const>()?;
    m.add_class::<InlineDataset>()?;
    m.add_class::<GradientDescent>()?;
    m.add_class::<BarrierSync>()?;
    m.add_class::<NonBlockingSync>()?;
    m.add_class::<BlockingStore>()?;
    m.add_class::<WildStore>()?;
    m.add_function(wrap_pyfunction!(parameter_server, m)?)?;
    m.add_function(wrap_pyfunction!(orchestrate, m)?)?;
    Ok(())
}
