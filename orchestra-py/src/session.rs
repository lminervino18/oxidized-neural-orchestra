use std::{
    io::{self, IsTerminal, Write},
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    },
    thread::{self, JoinHandle},
    time::Duration,
};

use orchestrator::sessions::{CancelHandle, TrainingEvent};
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use tokio::sync::mpsc::Receiver;

const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const BAR_WIDTH: usize = 40;

fn fmt_loss(loss: f64) -> String {
    if loss.abs() < 1e-4 {
        format!("{loss:.3e}")
    } else {
        format!("{loss:.8}")
    }
}

fn avg_loss<I>(losses: I) -> f64
where
    I: IntoIterator<Item = f64>,
{
    let mut sum = 0.0;
    let mut count = 0usize;

    for loss in losses.into_iter() {
        sum += loss;
        count += 1;
    }

    if count == 0 {
        return 0.0;
    }

    sum / count as f64
}

/// Tracks per-worker training progress and renders it to stdout.
///
/// In TTY mode renders an animated spinner with an in-place progress bar.
/// In non-TTY mode (pipes, CI) prints one line per epoch update instead.
struct ProgressReporter {
    worker_losses: Vec<Vec<f64>>,
    loss_history: Vec<f64>,
    max_epochs: usize,
    is_tty: bool,
    current_epoch: Arc<AtomicUsize>,
    avg_loss_bits: Arc<AtomicUsize>,
    done: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl ProgressReporter {
    fn new(max_epochs: usize, worker_count: usize) -> Self {
        let is_tty = io::stdout().is_terminal();

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

            Some(thread::spawn(move || {
                while !done.load(Ordering::Relaxed) {
                    let i = spinner_i.fetch_add(1, Ordering::Relaxed);
                    let spinner = SPINNER[i % SPINNER.len()];
                    let epoch = current_epoch.load(Ordering::Relaxed);
                    let loss = f64::from_bits(avg_loss_bits.load(Ordering::Relaxed) as u64);
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

                    let _ = io::stdout().flush();
                    thread::sleep(Duration::from_millis(80));
                }
            }))
        } else {
            None
        };

        Self {
            worker_losses: vec![vec![]; worker_count],
            loss_history: Vec::new(),
            max_epochs,
            is_tty,
            current_epoch,
            avg_loss_bits,
            done,
            handle,
        }
    }

    fn add_losses(&mut self, worker_id: usize, losses: &[f64]) {
        if worker_id < self.worker_losses.len() {
            self.worker_losses[worker_id].extend_from_slice(losses);
        }
    }

    fn update_avg_losses(&mut self) {
        loop {
            let epoch = self.loss_history.len();

            let maybe_epoch_losses = self
                .worker_losses
                .iter()
                .map(|wls| wls.get(epoch).copied())
                .collect::<Option<Vec<_>>>();

            let Some(epoch_losses) = maybe_epoch_losses else {
                break;
            };

            let avg_loss = avg_loss(epoch_losses);
            self.loss_history.push(avg_loss);

            let avg_loss_bits = avg_loss.to_bits() as usize;
            self.avg_loss_bits.store(avg_loss_bits, Ordering::Relaxed);
            self.current_epoch.store(epoch, Ordering::Relaxed);

            if !self.is_tty {
                println!(
                    "  epoch {}/{} avg_loss={}",
                    epoch,
                    self.max_epochs,
                    fmt_loss(avg_loss)
                );

                let _ = io::stdout().flush();
            }
        }
    }

    fn finish(self, success: bool) {
        let Self {
            done,
            handle,
            is_tty,
            max_epochs,
            loss_history,
            ..
        } = self;

        done.store(true, Ordering::Relaxed);

        if let Some(h) = handle {
            let _ = h.join();
        }

        if is_tty {
            let mark = if success { "✓" } else { "✗" };
            let epoch = loss_history.len();

            if let Some(&last_loss) = loss_history.last() {
                print!(
                    "\x1b[2A\r  {} [{}] {}/{}\n  avg_loss={}\n\n",
                    mark,
                    "█".repeat(BAR_WIDTH),
                    epoch,
                    max_epochs,
                    fmt_loss(last_loss),
                );

                let _ = io::stdout().flush();
            }
        }
    }
}

#[pyclass]
pub struct TrainedModel {
    pub inner: orchestrator::TrainedModel,
    pub loss_history: Vec<f64>,
}

#[pymethods]
impl TrainedModel {
    /// Returns the trained model parameters as a flat vector.
    ///
    /// # Args
    /// This method does not take arguments.
    ///
    /// # Returns
    /// The trained parameters in a flat vector.
    pub fn weights(&self) -> Vec<f32> {
        self.inner.params.to_vec()
    }

    /// Returns the average training loss recorded at each epoch.
    ///
    /// # Returns
    /// One loss value per completed epoch, in order.
    pub fn loss_history(&self) -> Vec<f64> {
        self.loss_history.clone()
    }

    /// Saves the trained model in safetensors format.
    ///
    /// # Args
    /// * `path` - Destination path for the safetensors file.
    ///
    /// # Returns
    /// `None`.
    pub fn save_safetensors(&self, path: &str) -> PyResult<()> {
        self.inner
            .save_safetensors(path)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
}

#[pyclass]
pub struct Session {
    pub inner: Option<(orchestrator::Session, Receiver<()>)>,
    pub cancel: CancelHandle,
    pub max_epochs: usize,
    pub worker_count: usize,
}

#[pymethods]
impl Session {
    /// Requests an orderly stop of the training session at the next epoch boundary.
    ///
    /// If the session was already consumed or already stopping, this is a no-op.
    pub fn stop(&self) {
        self.cancel.stop();
    }

    /// Blocks until training completes and returns the trained model.
    ///
    /// # Returns
    /// The trained model with its final parameters.
    ///
    /// # Errors
    /// Raises a `RuntimeError` if the session was already consumed, if training
    /// fails, or if the background thread panics.
    pub fn wait(mut slf: PyRefMut<'_, Self>, py: Python<'_>) -> PyResult<TrainedModel> {
        let (session, cancel_rx) = slf
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("session already consumed"))?;

        let max_epochs = slf.max_epochs;
        let worker_count = slf.worker_count;

        drop(slf);

        let trained = py
            .detach(|| {
                thread::spawn(move || {
                    let mut rx = session.event_listener(cancel_rx);
                    let mut reporter = ProgressReporter::new(max_epochs, worker_count);

                    let result = loop {
                        match rx.blocking_recv() {
                            Some(TrainingEvent::PublishedLosses { worker_id, losses }) => {
                                reporter.add_losses(worker_id, &losses);
                                reporter.update_avg_losses();
                            }
                            Some(TrainingEvent::TrainingComplete { model: trained, .. }) => {
                                break Ok(trained)
                            }
                            Some(TrainingEvent::Error(e)) => break Err(e.to_string()),
                            Some(_) => continue,
                            None => break Err("session channel closed unexpectedly".into()),
                        }
                    };

                    let loss_history = reporter.loss_history.clone();
                    reporter.finish(result.is_ok());
                    result.map(|trained| (trained, loss_history))
                })
                .join()
                .map_err(|_| "session thread panicked".to_string())?
            })
            .map_err(PyRuntimeError::new_err)?;

        let (trained, loss_history) = trained;
        let trained_model = TrainedModel {
            inner: trained,
            loss_history,
        };

        Ok(trained_model)
    }
}
