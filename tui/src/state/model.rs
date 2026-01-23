use std::time::{Duration, Instant};

/// High-level lifecycle states for a training session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionPhase {
    Init,
    Connecting,
    Training,
    Finished,
    Error,
}

/// Worker runtime status shown in the TUI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerStatus {
    WaitingWeights,
    Computing,
    SendingGradients,
    Disconnected,
    Error,
}

/// Immutable worker metadata + live status.
#[derive(Debug, Clone)]
pub struct WorkerView {
    pub worker_id: usize,
    pub step: usize,
    pub steps_total: usize,
    pub strategy_kind: &'static str,
    pub status: WorkerStatus,
}

/// Parameter Server view.
#[derive(Debug, Clone)]
pub struct ServerView {
    pub trainer_kind: &'static str,
    pub optimizer_kind: &'static str,
    pub shard_size: usize,
    pub num_params: usize,
}

/// A single log entry shown in the event panel.
#[derive(Debug, Clone)]
pub struct LogLine {
    pub level: &'static str,
    pub message: String,
}

/// Full snapshot rendered by the TUI.
#[derive(Debug, Clone)]
pub struct SessionView {
    pub phase: SessionPhase,
    pub started_at: Instant,
    pub elapsed: Duration,
    pub step_done: usize,
    pub step_total: usize,
    pub workers_connected: usize,
    pub workers_total: usize,
    pub server: ServerView,
    pub workers: Vec<WorkerView>,
    pub logs: Vec<LogLine>,
}
