use std::time::{Duration, Instant};

use super::model::{
    LogLine, ServerView, SessionPhase, SessionView, WorkerStatus, WorkerView,
};

/// Mock state provider for the TUI.
///
/// Today: hardcoded + animated with `tick()`.
/// Tomorrow: replace with real orchestrator telemetry/state.
#[derive(Debug)]
pub struct MockState {
    started_at: Instant,
    tick: usize,
    view: SessionView,
}

impl MockState {
    /// Creates a new mock session view.
    ///
    /// # Returns
    /// A mock state instance.
    pub fn new() -> Self {
        let started_at = Instant::now();

        let workers_total = 3;
        let step_total = 500;

        let server = ServerView {
            trainer_kind: "barrier_sync",
            optimizer_kind: "adam(lr=0.1,b1=0.2,b2=0.2,eps=0.01)",
            shard_size: 128,
            num_params: 2048,
        };

        let workers = (1..=workers_total)
            .map(|id| WorkerView {
                worker_id: id,
                step: 0,
                steps_total: step_total,
                strategy_kind: "mock",
                status: WorkerStatus::WaitingWeights,
            })
            .collect::<Vec<_>>();

        let logs = vec![
            LogLine {
                level: "INFO",
                message: "orchestrator initialized".into(),
            },
            LogLine {
                level: "INFO",
                message: "connected to parameter server".into(),
            },
        ];

        let view = SessionView {
            phase: SessionPhase::Connecting,
            started_at,
            elapsed: Duration::from_secs(0),
            step_done: 0,
            step_total,
            workers_connected: 0,
            workers_total,
            server,
            workers,
            logs,
        };

        Self {
            started_at,
            tick: 0,
            view,
        }
    }

    /// Returns a snapshot used by the UI.
    pub fn view(&self) -> SessionView {
        self.view.clone()
    }

    /// Advances the mock simulation.
    pub fn tick(&mut self) {
        self.tick += 1;
        self.view.elapsed = self.started_at.elapsed();

        // Phase progression
        if self.tick < 10 {
            self.view.phase = SessionPhase::Connecting;
            self.view.workers_connected = (self.tick / 3).min(self.view.workers_total);
        } else if self.tick < 400 {
            self.view.phase = SessionPhase::Training;
            self.view.workers_connected = self.view.workers_total;

            // Steps advance
            self.view.step_done = (self.view.step_done + 1).min(self.view.step_total);

            // Per-worker animation: staggered steps and rotating statuses
            for (idx, w) in self.view.workers.iter_mut().enumerate() {
                let lag = idx; // stagger by index
                if self.view.step_done > lag {
                    w.step = (self.view.step_done - lag).min(w.steps_total);
                }

                w.status = match (self.tick + idx) % 3 {
                    0 => WorkerStatus::WaitingWeights,
                    1 => WorkerStatus::Computing,
                    _ => WorkerStatus::SendingGradients,
                };
            }

            if self.view.step_done % 25 == 0 {
                self.view.logs.push(LogLine {
                    level: "INFO",
                    message: format!("barrier reached at step {}", self.view.step_done),
                });
                self.trim_logs();
            }
        } else {
            self.view.phase = SessionPhase::Finished;
            self.view.step_done = self.view.step_total;
            for w in self.view.workers.iter_mut() {
                w.step = w.steps_total;
                w.status = WorkerStatus::WaitingWeights;
            }
            if self.tick == 400 {
                self.view.logs.push(LogLine {
                    level: "INFO",
                    message: "training finished, weights ready".into(),
                });
                self.trim_logs();
            }
        }
    }

    fn trim_logs(&mut self) {
        const MAX: usize = 200;
        if self.view.logs.len() > MAX {
            let drain = self.view.logs.len() - MAX;
            self.view.logs.drain(0..drain);
        }
    }
}
