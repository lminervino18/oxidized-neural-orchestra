use comms::{NetRtp, WorkerHandle};
use tokio::sync::mpsc::{Receiver, Sender};

use crate::TrainingEvent;

/// Requests that the orchestrator can make to a worker handler task.
#[derive(Debug)]
pub enum WorkerRequest {
    PullParams,
    Disconnect,
    Stop,
}

pub trait TrainingStrategy {
    /// Spawns a worker handler task for a worker.
    ///
    /// # Args
    /// * `id` - The worker's id.
    /// * `worker_handle` - The handle for communicating with the worker.
    /// * `rx` - The worker's receiver for communicating worker requests from the orchestrator.
    /// * `tx` - The worker's sender for communicating training events to the orchestrator.
    ///
    /// # Returns
    /// An orch error if occurred.
    async fn spawn(
        id: usize,
        worker_handle: WorkerHandle<NetRtp>,
        rx: Receiver<WorkerRequest>,
        tx: Sender<TrainingEvent>,
    );
}
