use std::io::{IsTerminal, Write};
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};
use std::thread::JoinHandle;

use orchestrator::TrainingEvent;
use pyo3::prelude::*;

const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const BAR_WIDTH: usize = 40;

fn fmt_loss(loss: f32) -> String {
    if loss.abs() < 1e-4 {
        format!("{loss:.3e}")
    } else {
        format!("{loss:.8}")
    }
}

/// Tracks per-worker training progress and renders it to stdout.
///
/// In TTY mode renders an animated spinner with an in-place progress bar.
/// In non-TTY mode (pipes, CI) prints one line per epoch update instead.
struct ProgressReporter {
    worker_epochs: Vec<usize>,
    last_loss: Vec<Option<f32>>,
    max_epochs: usize,
    is_tty: bool,
    current_epoch: Arc<AtomicUsize>,
    avg_loss_bits: Arc<AtomicUsize>,
    done: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl ProgressReporter {
    fn new(max_epochs: usize, worker_count: usize) -> Self {
        let is_tty = std::io::stdout().is_terminal();

        let current_epoch = Arc::new(AtomicUsize::new(0));
        let avg_loss_bits = Arc::new(AtomicUsize::new(0f32.to_bits() as usize));
        let done = Arc::new(AtomicBool::new(false));
        let spinner_i = Arc::new(AtomicUsize::new(0));

        let handle = if is_tty {
            println!();
            println!();

            let current_epoch = Arc::clone(&current_epoch);
            let avg_loss_bits = Arc::clone(&avg_loss_bits);
            let done = Arc::clone(&done);

            Some(std::thread::spawn(move || {
                while !done.load(Ordering::Relaxed) {
                    let i = spinner_i.fetch_add(1, Ordering::Relaxed);
                    let spinner = SPINNER[i % SPINNER.len()];
                    let epoch = current_epoch.load(Ordering::Relaxed);
                    let loss = f32::from_bits(avg_loss_bits.load(Ordering::Relaxed) as u32);
                    let filled = ((epoch * BAR_WIDTH) / max_epochs.max(1)).min(BAR_WIDTH);

                    print!(
                        "\x1b[2A\r  {} [{}{}] {}/{}\n  avg_loss={}\n",
                        spinner,
                        "█".repeat(filled),
                        "░".repeat(BAR_WIDTH - filled),
                        epoch,
                        max_epochs,
                        fmt_loss(loss),
                    );
                    let _ = std::io::stdout().flush();
                    std::thread::sleep(std::time::Duration::from_millis(80));
                }
            }))
        } else {
            None
        };

        Self {
            worker_epochs: vec![0; worker_count],
            last_loss: vec![None; worker_count],
            max_epochs,
            is_tty,
            current_epoch,
            avg_loss_bits,
            done,
            handle,
        }
    }

    /// Records losses for `worker_id` and refreshes the display.
    fn update(&mut self, worker_id: usize, losses: &[f32]) {
        for &loss in losses {
            if worker_id < self.worker_epochs.len() {
                self.worker_epochs[worker_id] += 1;
                self.last_loss[worker_id] = Some(loss);
            }
        }

        let epoch = self.worker_epochs.iter().copied().max().unwrap_or(0);
        let avg = self.avg_loss();

        self.current_epoch.store(epoch, Ordering::Relaxed);
        self.avg_loss_bits
            .store(avg.to_bits() as usize, Ordering::Relaxed);

        if !self.is_tty {
            println!(
                "  epoch {}/{} avg_loss={}",
                epoch,
                self.max_epochs,
                fmt_loss(avg)
            );
            let _ = std::io::stdout().flush();
        }
    }

    /// Stops the spinner and prints the final summary line.
    fn finish(self, success: bool) {
        self.done.store(true, Ordering::Relaxed);

        if let Some(handle) = self.handle {
            let _ = handle.join();
        }

        if self.is_tty {
            let avg_loss = self.avg_loss();
            let epoch = if success {
                self.max_epochs
            } else {
                self.worker_epochs.iter().copied().max().unwrap_or(0)
            };
            let mark = if success { "✓" } else { "✗" };

            print!(
                "\x1b[2A\r  {} [{}] {}/{}\n  avg_loss={}\n\n",
                mark,
                "█".repeat(BAR_WIDTH),
                epoch,
                self.max_epochs,
                fmt_loss(avg_loss),
            );
            let _ = std::io::stdout().flush();
        }
    }

    fn avg_loss(&self) -> f32 {
        let reported: Vec<f32> = self.last_loss.iter().filter_map(|l| *l).collect();

        if reported.is_empty() {
            return 0.0;
        }

        reported.iter().sum::<f32>() / reported.len() as f32
    }
}

#[pyclass]
pub struct TrainedModel {
    pub inner: orchestrator::TrainedModel,
}

#[pymethods]
impl TrainedModel {
    pub fn weights(&self) -> Vec<f32> {
        self.inner.params().to_vec()
    }

    pub fn save_safetensors(&self, path: &str) -> PyResult<()> {
        self.inner
            .save_safetensors(path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }
}

#[pyclass]
pub struct Session {
    pub inner: Option<orchestrator::Session>,
    pub max_epochs: usize,
    pub worker_count: usize,
}

#[pymethods]
impl Session {
    /// Blocks until training completes and returns the trained model.
    ///
    /// # Returns
    /// The trained model with its final parameters.
    ///
    /// # Errors
    /// Raises a `RuntimeError` if the session was already consumed, if training
    /// fails, or if the background thread panics.
    pub fn wait(&mut self, py: Python<'_>) -> PyResult<TrainedModel> {
        let session = self
            .inner
            .take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("session already consumed"))?;

        let max_epochs = self.max_epochs;
        let worker_count = self.worker_count;

        let trained = py
            .allow_threads(|| {
                std::thread::spawn(move || {
                    let mut rx = session.event_listener();
                    let mut reporter = ProgressReporter::new(max_epochs, worker_count);

                    let result = loop {
                        match rx.blocking_recv() {
                            Some(TrainingEvent::Loss { worker_id, losses }) => {
                                reporter.update(worker_id, &losses);
                            }
                            Some(TrainingEvent::Complete(trained)) => break Ok(trained),
                            Some(TrainingEvent::Error(e)) => break Err(e.to_string()),
                            Some(_) => continue,
                            None => break Err("session channel closed unexpectedly".into()),
                        }
                    };

                    reporter.finish(result.is_ok());
                    result
                })
                .join()
                .map_err(|_| "session thread panicked".to_string())?
            })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        Ok(TrainedModel { inner: trained })
    }
}