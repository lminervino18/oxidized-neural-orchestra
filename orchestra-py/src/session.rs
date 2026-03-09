use std::io::Write;

use orchestrator::TrainingEvent;
use pyo3::prelude::*;

const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

/// The final trained model returned after a session completes.
#[pyclass]
pub struct TrainedModel {
    pub params: Vec<f32>,
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
    /// * `output_sizes` - List of output sizes per layer, e.g. `[8, 4, 1]`.
    /// * `input_size` - Input size of the first layer.
    ///
    /// # Errors
    /// Raises a `RuntimeError` if the file cannot be written.
    pub fn save(&self, path: &str, output_sizes: Vec<usize>, input_size: usize) -> PyResult<()> {
        let mut csv = String::from("layer,type,index,value\n");
        let mut offset = 0;
        let mut prev = input_size;

        for (layer_i, &out) in output_sizes.iter().enumerate() {
            let w_count = prev * out;
            let b_count = out;
            for i in 0..w_count {
                csv.push_str(&format!("{layer_i},weight,{i},{}\n", self.params[offset + i]));
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

/// A handle to an ongoing training session.
#[pyclass]
pub struct Session {
    pub inner: Option<orchestrator::Session>,
    pub max_epochs: usize,
    pub worker_count: usize,
}

#[pymethods]
impl Session {
    /// Blocks until training completes, showing a progress bar.
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
                                let avg_loss =
                                    reported.iter().sum::<f32>() / reported.len() as f32;
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