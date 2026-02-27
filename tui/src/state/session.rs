use std::time::Instant;

use orchestrator::TrainingEvent;
use tokio::sync::mpsc;

use super::model::{LogLine, SessionPhase, SessionView, WorkerStatus, WorkerView};

const MAX_LOGS: usize = 200;

/// Drives the TUI state from a stream of [`TrainingEvent`]s.
pub struct SessionState {
    view: SessionView,
    events: mpsc::Receiver<TrainingEvent>,
}

impl SessionState {
    /// Creates a new `SessionState` for the given number of workers.
    ///
    /// # Args
    /// * `workers_total` - Expected number of workers in this session.
    /// * `events` - The receiver end of the training events channel.
    pub fn new(workers_total: usize, events: mpsc::Receiver<TrainingEvent>) -> Self {
        let workers = (0..workers_total)
            .map(|id| WorkerView {
                worker_id: id,
                losses: Vec::new(),
                status: WorkerStatus::Active,
            })
            .collect();

        let view = SessionView {
            phase: SessionPhase::Connecting,
            started_at: Instant::now(),
            elapsed: Default::default(),
            workers_total,
            workers_done: 0,
            workers,
            final_params: None,
            logs: vec![LogLine {
                level: "INFO",
                message: "connecting to workers and parameter server...".into(),
            }],
        };

        Self { view, events }
    }

    /// Returns the current snapshot for rendering.
    pub fn view(&self) -> SessionView {
        self.view.clone()
    }

    /// Drains all pending events and updates state. Non-blocking.
    ///
    /// Should be called once per TUI frame tick.
    pub fn tick(&mut self) {
        self.view.elapsed = self.view.started_at.elapsed();

        // Drain all events that are ready right now without blocking.
        while let Ok(event) = self.events.try_recv() {
            self.apply(event);
        }
    }

    fn apply(&mut self, event: TrainingEvent) {
        match event {
            TrainingEvent::Loss { worker_id, losses } => {
                if let Some(w) = self.view.workers.get_mut(worker_id) {
                    let epoch = w.losses.len() + 1;
                    let last = losses.last().copied();
                    w.losses.extend(losses);
                    if let Some(loss) = last {
                        self.push_log(
                            "INFO",
                            format!("worker {worker_id} epoch {epoch}: loss={loss:.4}"),
                        );
                    }
                }
                self.view.phase = SessionPhase::Training;
            }

            TrainingEvent::WorkerDone(worker_id) => {
                if let Some(w) = self.view.workers.get_mut(worker_id) {
                    w.status = WorkerStatus::Disconnected;
                }
                self.view.workers_done += 1;
                self.push_log("INFO", format!("worker {worker_id} disconnected"));
            }

            TrainingEvent::Complete(params) => {
                self.view.phase = SessionPhase::Finished;
                self.push_log(
                    "INFO",
                    format!("training complete — {} parameters received", params.len()),
                );
                self.view.final_params = Some(params);
            }

            TrainingEvent::Error(msg) => {
                self.view.phase = SessionPhase::Error;
                self.push_log("ERROR", msg);
            }
        }
    }

    fn push_log(&mut self, level: &'static str, message: String) {
        self.view.logs.push(LogLine { level, message });
        if self.view.logs.len() > MAX_LOGS {
            let drain = self.view.logs.len() - MAX_LOGS;
            self.view.logs.drain(0..drain);
        }
    }
}