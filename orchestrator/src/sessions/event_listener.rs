use std::mem;

use comms::{NetRtp, ParamServerHandle};
use log::info;
use tokio::sync::mpsc::{Receiver, Sender};

use crate::{
    StopReason, TrainingEvent,
    configs::{StrategySwitchTracking, WorkerPostAction},
    sessions::{ConvergenceTracker, LossRecorder, WorkerRequest},
};

/// The main loop over the training events in the system.
pub struct EventListener<'a> {
    cancel_rx: Receiver<()>,
    server_handles: &'a mut Vec<ParamServerHandle<NetRtp>>,
    req_txs: &'a mut [Sender<WorkerRequest>],
    event_rx: &'a mut Receiver<TrainingEvent>,
    event_tx: Sender<TrainingEvent>,

    // Training state
    workers_left: usize,
    loss_recorder: LossRecorder,
    switch_tracking: Option<StrategySwitchTracking>,
    convergence_tracker: Option<ConvergenceTracker>,
    stop_reason: Option<StopReason>,
}

impl<'a> EventListener<'a> {
    /// Creates a new `EventListener`.
    ///
    /// # Args
    /// * `cancel_rx` - The training cancellation request receiver.
    /// * `req_txs` - The request senders for the worker listeners.
    /// * `server_handles` - The server handles session vec.
    /// * `loss_recorder` - The workers' loss recorder.
    /// * `convergence_tracker` - A tracker device to track model convergence.
    /// * `event_rx` - An event producer.
    /// * `event_tx` - An event consumer.
    /// * `switch_tracking` - The strategy switch tracking metadata.
    ///
    /// # Returns
    /// A new `EventListener` instance.
    pub fn new(
        cancel_rx: Receiver<()>,
        req_txs: &'a mut [Sender<WorkerRequest>],
        server_handles: &'a mut Vec<ParamServerHandle<NetRtp>>,
        loss_recorder: LossRecorder,
        convergence_tracker: Option<ConvergenceTracker>,
        event_rx: &'a mut Receiver<TrainingEvent>,
        event_tx: Sender<TrainingEvent>,
        switch_tracking: Option<StrategySwitchTracking>,
    ) -> Self {
        let nworkers = req_txs.len();

        Self {
            cancel_rx,
            server_handles,
            req_txs,
            loss_recorder,
            convergence_tracker,
            event_rx,
            event_tx,
            switch_tracking,
            workers_left: nworkers,
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
                self.handle_losses(worker_id, &losses).await;
                let event = TrainingEvent::PublishedLosses { worker_id, losses };
                let _ = self.event_tx.send(event).await;
                Some(true)
            }
            TrainingEvent::Upgrade { server_handle } => {
                self.server_handles.push(*server_handle);
                Some(true)
            }
            other => {
                let is_err = matches!(other, TrainingEvent::Error(..));
                let _ = self.event_tx.send(other).await;
                (!is_err).then_some(true)
            }
        }
    }

    /// Handles the latest loss update from a worker.
    ///
    /// # Args
    /// * `worker_id` - The id of the worker whose losses are being processed.
    /// * `losses` - The losses it published.
    async fn handle_losses(&mut self, worker_id: usize, losses: &[f64]) {
        if self.stop_reason.is_some() {
            return;
        }

        if let Some(&last) = losses.last() {
            self.loss_recorder.record(worker_id, last);
        }

        let Some(loss) = self.loss_recorder.max() else {
            return;
        };

        self.loss_recorder.clear();

        if let Some(ref mut tracker) = self.convergence_tracker {
            tracker.record(loss);

            if tracker.converged() {
                info!("early stopping triggered");
                self.stop_reason = Some(StopReason::EarlyStopping);
                self.broadcast_request(WorkerRequest::Stop).await;
                return;
            }
        }

        if let Some(ref mut switch_tracking) = self.switch_tracking {
            let StrategySwitchTracking {
                tracker,
                post_actions,
            } = switch_tracking;

            tracker.record(loss);

            if tracker.should_switch() {
                info!("strategy switch triggered");
                let taken = mem::take(post_actions);
                self.broadcast_switch(taken).await;
                self.switch_tracking = None;
            }
        }
    }

    /// Broadcasts the `WorkerPostActions` to all the workers.
    ///
    /// # Args
    /// * `post_actios` - The action that each worker needs to take in order to switch algorithm.
    async fn broadcast_switch(&mut self, post_actions: Vec<WorkerPostAction>) {
        for (tx, action) in self.req_txs.iter_mut().zip(post_actions) {
            let req = match action {
                WorkerPostAction::Switch {
                    server_addrs,
                    server_sizes,
                    server_ordering,
                } => WorkerRequest::Switch {
                    server_addrs,
                    server_sizes,
                    server_ordering,
                },
                WorkerPostAction::Upgrade { spec, ranges } => WorkerRequest::Upgrade {
                    spec: Box::new(spec),
                    ranges,
                },
            };

            let _ = tx.send(req).await;
        }
    }

    /// Broadcasts a single request to all of the request senders.
    ///
    /// # Args
    /// * `req` - The request to broadcast.
    async fn broadcast_request(&mut self, req: WorkerRequest) {
        for tx in self.req_txs.iter_mut() {
            let _ = tx.send(req.clone()).await;
        }
    }
}
