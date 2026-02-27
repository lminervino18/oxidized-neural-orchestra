use std::time::{Duration, Instant};

/// High-level lifecycle states for a training session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionPhase {
    Connecting,
    Training,
    Finished,
    Error,
}

/// Live state for a single worker.
#[derive(Debug, Clone)]
pub struct WorkerView {
    pub worker_id: usize,
    /// Loss history reported by this worker across epochs.
    pub losses: Vec<f32>,
    pub status: WorkerStatus,
}

/// Worker runtime status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerStatus {
    Active,
    Disconnected,
    Error,
}

/// A single log entry.
#[derive(Debug, Clone)]
pub struct LogLine {
    pub level: &'static str,
    pub message: String,
}

/// Full snapshot rendered by the TUI each frame.
#[derive(Debug, Clone)]
pub struct SessionView {
    pub phase: SessionPhase,
    pub started_at: Instant,
    pub elapsed: Duration,
    pub workers_total: usize,
    pub workers_done: usize,
    pub workers: Vec<WorkerView>,
    pub final_params: Option<Vec<f32>>,
    pub logs: Vec<LogLine>,
}