use std::io::{IsTerminal, Write};
use std::sync::{Arc, atomic::{AtomicBool, AtomicUsize, Ordering}};

use orchestrator::TrainingEvent;
use pyo3::prelude::*;

const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

fn fmt_loss(loss: f32) -> String {
    if loss.abs() < 1e-4 {
        format!("{loss:.3e}")
    } else {
        format!("{loss:.8}")
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
    pub fn wait(&mut self, py: Python<'_>) -> PyResult<TrainedModel> {
        let session = self
            .inner
            .take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("session already consumed"))?;

        let max_epochs = self.max_epochs;
        let worker_count = self.worker_count;
        let is_tty = std::io::stdout().is_terminal();

        let trained = py
            .allow_threads(|| {
                std::thread::spawn(move || {
                    let mut rx = session.event_listener();
                    let mut worker_epochs: Vec<usize> = vec![0; worker_count];
                    let mut last_loss: Vec<Option<f32>> = vec![None; worker_count];
                    let bar_width = 40usize;

                    // Shared state for the spinner thread
                    let spinner_i = Arc::new(AtomicUsize::new(0));
                    let current_epoch = Arc::new(AtomicUsize::new(0));
                    let avg_loss_bits = Arc::new(AtomicUsize::new(0f32.to_bits() as usize));
                    let done = Arc::new(AtomicBool::new(false));

                    // Spawn spinner thread only for TTY
                    let spinner_handle = if is_tty {
                        println!();
                        println!();

                        let spinner_i = Arc::clone(&spinner_i);
                        let current_epoch = Arc::clone(&current_epoch);
                        let avg_loss_bits = Arc::clone(&avg_loss_bits);
                        let done = Arc::clone(&done);

                        Some(std::thread::spawn(move || {
                            while !done.load(Ordering::Relaxed) {
                                let i = spinner_i.fetch_add(1, Ordering::Relaxed);
                                let spinner = SPINNER[i % SPINNER.len()];
                                let epoch = current_epoch.load(Ordering::Relaxed);
                                let loss = f32::from_bits(avg_loss_bits.load(Ordering::Relaxed) as u32);
                                let filled = ((epoch * bar_width) / max_epochs.max(1)).min(bar_width);

                                print!(
                                    "\x1b[2A\r  {} [{}{}] {}/{}\n  avg_loss={}\n",
                                    spinner,
                                    "█".repeat(filled),
                                    "░".repeat(bar_width - filled),
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

                    let result = loop {
                        match rx.blocking_recv() {
                            Some(TrainingEvent::Loss { worker_id, losses }) => {
                                for loss in &losses {
                                    if worker_id < worker_epochs.len() {
                                        worker_epochs[worker_id] += 1;
                                        last_loss[worker_id] = Some(*loss);
                                    }
                                }
                                let epoch = *worker_epochs.iter().max().unwrap_or(&0);
                                let reported: Vec<f32> =
                                    last_loss.iter().filter_map(|l| *l).collect();
                                let avg = reported.iter().sum::<f32>() / reported.len() as f32;

                                current_epoch.store(epoch, Ordering::Relaxed);
                                avg_loss_bits.store(avg.to_bits() as usize, Ordering::Relaxed);

                                if !is_tty {
                                    println!("  epoch {}/{} avg_loss={}", epoch, max_epochs, fmt_loss(avg));
                                    let _ = std::io::stdout().flush();
                                }
                            }
                            Some(TrainingEvent::Complete(trained)) => {
                                break Ok(trained);
                            }
                            Some(TrainingEvent::Error(e)) => {
                                break Err(e.to_string());
                            }
                            Some(_) => continue,
                            None => break Err("session channel closed unexpectedly".into()),
                        }
                    };

                    // Stop spinner thread
                    done.store(true, Ordering::Relaxed);
                    if let Some(handle) = spinner_handle {
                        let _ = handle.join();
                    }

                    // Print final state
                    if is_tty {
                        let reported: Vec<f32> = last_loss.iter().filter_map(|l| *l).collect();
                        let avg_loss = if reported.is_empty() {
                            0.0
                        } else {
                            reported.iter().sum::<f32>() / reported.len() as f32
                        };
                        let (mark, epoch) = if result.is_ok() { ("✓", max_epochs) } else { ("✗", *worker_epochs.iter().max().unwrap_or(&0)) };
                        print!(
                            "\x1b[2A\r  {} [{}{}] {}/{}\n  avg_loss={}\n\n",
                            mark,
                            "█".repeat(bar_width),
                            "",
                            epoch,
                            max_epochs,
                            fmt_loss(avg_loss),
                        );
                        let _ = std::io::stdout().flush();
                    }

                    result
                })
                .join()
                .map_err(|_| "session thread panicked".to_string())?
            })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        Ok(TrainedModel { inner: trained })
    }
}