use log::info;
use tokio::sync::mpsc::{Receiver, Sender};

use crate::{
    StopReason, TrainingEvent,
    configs::{EarlyStoppingConfig, WorkerPostAction},
    sessions::{ConvergenceTracker, WorkerRequest},
};

/// The main loop over the training events in the system.
pub struct EventListener<'a> {
    cancel_rx: Receiver<()>,
    req_txs: &'a mut [Sender<WorkerRequest>],
    early_stopping: Option<EarlyStoppingConfig>,
    event_rx: &'a mut Receiver<TrainingEvent>,
    event_tx: Sender<TrainingEvent>,
    worker_post_actions: Option<Vec<WorkerPostAction>>,

    // Training state
    workers_left: usize,
    tracker: ConvergenceTracker,
    stop_reason: Option<StopReason>,
}

impl<'a> EventListener<'a> {
    /// Creates a new `EventListener`.
    ///
    /// # Args
    /// * `cancel_rx` - The training cancellation request receiver.
    /// * `req_txs` - The request senders for the worker listeners.
    /// * `early_stopping` - The early stopping mechanism.
    /// * `event_rx` - An event producer.
    /// * `event_tx` - An event consumer.
    /// * `worker_post_actions` - The actions to take per worker for strategy switch.
    ///
    /// # Returns
    /// A new `EventListener` instance.
    pub fn new(
        cancel_rx: Receiver<()>,
        req_txs: &'a mut [Sender<WorkerRequest>],
        early_stopping: Option<EarlyStoppingConfig>,
        event_rx: &'a mut Receiver<TrainingEvent>,
        event_tx: Sender<TrainingEvent>,
        worker_post_actions: Option<Vec<WorkerPostAction>>,
    ) -> Self {
        let n_workers = req_txs.len();

        Self {
            cancel_rx,
            req_txs,
            early_stopping,
            event_rx,
            event_tx,
            worker_post_actions,
            workers_left: n_workers,
            tracker: ConvergenceTracker::new(n_workers),
            stop_reason: None,
        }
    }

    /// The main loop over the events of the system and the user. It listens
    /// for training events coming from the workers and takes action.
    ///
    /// # Returns
    /// An optional training `StopReason`.
    pub async fn listen(&mut self) -> Option<StopReason> {
        let mut should_continue = true;

        while should_continue {
            tokio::select! {
                biased;
                _ = self.cancel_rx.recv(), if self.stop_reason.is_none() => {
                    info!("manual stop requested by the user");
                    self.stop_reason = Some(StopReason::ManualStop);
                    self.broadcast_request(WorkerRequest::Stop).await;
                }
                event = self.event_rx.recv() => {
                    let Some(event) = event else {
                        break;
                    };

                    should_continue = self.handle_event(event).await?;
                }
            }
        }

        Some(self.stop_reason.unwrap_or_default())
    }

    /// Broadcasts a single request to all of the request senders.
    ///
    /// # Args
    /// * `req` - The request to broadcast.
    async fn broadcast_request(&mut self, req: WorkerRequest) {
        for tx in self.req_txs.iter_mut() {
            let _ = tx.send(req).await;
        }
    }

    /// Handles an incoming event from a worker.
    ///
    /// # Args
    /// * `event` - The incoming event.
    ///
    /// # Returns
    /// A should continue flag. If `false` the listener should halt.
    async fn handle_event(&mut self, event: TrainingEvent) -> Option<bool> {
        match event {
            TrainingEvent::WorkerDone(id) => {
                let _ = self.event_tx.send(TrainingEvent::WorkerDone(id)).await;
                self.workers_left = self.workers_left.saturating_sub(1);
                Some(self.workers_left > 0)
            }
            TrainingEvent::PublishedLosses { worker_id, losses } => {
                // TODO: Chequear aca si se cumple la condicion para hacer la transicion de strategy switch

                if self.stop_reason.is_none()
                    && let Some(ref cfg) = self.early_stopping
                    && let Some((prev, curr)) = self.tracker.record(worker_id, &losses)
                    && (prev - curr).abs() < *cfg.tolerance as f64
                {
                    info!("early stopping triggered (prev={prev:.6}, curr={curr:.6})");
                    self.stop_reason = Some(StopReason::EarlyStopping);
                    self.broadcast_request(WorkerRequest::Stop).await;
                }

                let event = TrainingEvent::PublishedLosses { worker_id, losses };
                let _ = self.event_tx.send(event).await;
                Some(true)
            }
            other => {
                let is_err = matches!(other, TrainingEvent::Error(..));
                let _ = self.event_tx.send(other).await;
                (!is_err).then_some(true)
            }
        }
    }
}
