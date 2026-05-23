use comms::{NetRtp, WorkerEvent, WorkerHandle, specs::server::ServerSpec};
use log::{debug, error, info, warn};
use tokio::sync::mpsc::{Receiver, Sender};

use super::{TrainingEvent, WorkerRequest};
use crate::{OrchErr, Result};

/// The return type of the `handle_request` method.
enum ReqResolution {
    Upgrade {
        spec: ServerSpec,
        ranges: Vec<(usize, usize)>,
    },
    Continue,
    Halt,
}

/// The return type of the `handle_event` method.
enum EventResolution {
    NotifyOrch(TrainingEvent),
    Exit,
}

/// The worker handle manager.
pub struct WorkerListener {
    id: usize,
    worker_handle: WorkerHandle<NetRtp>,
    stopping: bool,
}

impl WorkerListener {
    /// Creates a new `WorkerListener`.
    ///
    /// # Args
    /// * `id` - The id of the worker being requested.
    /// * `worker_handle` - The worker's communication handle.
    ///
    /// # Returns
    /// A new `WorkerListener` instance.
    pub fn new(id: usize, worker_handle: WorkerHandle<NetRtp>) -> Self {
        Self {
            id,
            worker_handle,
            stopping: false,
        }
    }

    /// Starts the worker listener.
    ///
    /// # Args
    /// * `req_rx` - The request receiver.
    /// * `event_tx` - The training events sender.
    ///
    /// # Returns
    /// An orch error if occurred.
    pub async fn listen(
        mut self,
        mut req_rx: Receiver<WorkerRequest>,
        event_tx: Sender<TrainingEvent>,
    ) {
        let id = self.id;

        loop {
            tokio::select! {
                req = req_rx.recv() => {
                    if let Some(req) = req {
                        match self.handle_request(req, &event_tx).await {
                            Ok(ReqResolution::Continue) => continue,
                            Ok(ReqResolution::Halt) => {},
                            Ok(ReqResolution::Upgrade { spec, ranges }) => {
                                let event = match self.worker_handle.upgrade(spec, ranges).await {
                                    Ok(server_handle) => TrainingEvent::Upgrade { server_handle },
                                    Err(e) => {
                                        let err = OrchErr::WorkerError { id, details: e.to_string() };
                                        TrainingEvent::Error(err)
                                    }
                                };

                                let _ = event_tx.send(event).await;
                            },
                            Err(e) => { let _ = event_tx.send(TrainingEvent::Error(e)).await; }
                        }

                        break;
                    }
                },
                event = self.worker_handle.recv_event() => match event {
                    Ok(event) => match Self::handle_event(id, event) {
                        Ok(EventResolution::Exit) => break,
                        Ok(EventResolution::NotifyOrch(event)) => {
                            let _ = event_tx.send(event).await;
                        }
                        Err(e) => {
                            let _ = event_tx.send(TrainingEvent::Error(e)).await;
                            break;
                        }
                    }
                    Err(e) => {
                        error!("worker {id} error: {e}");
                        let err = OrchErr::WorkerError { id, details: e.to_string() };
                        let _ = event_tx.send(TrainingEvent::Error(err)).await;
                        break;
                    }
                }
            }
        }
    }

    /// Handles the orchestrator's requests for a worker.
    ///
    /// # Args
    /// * `req` - The request that was made.
    /// * `event_tx` - The event producer for the orchestrator.
    ///
    /// # Returns
    /// A should continue flag or an orch error if occurred.
    async fn handle_request(
        &mut self,
        req: WorkerRequest,
        event_tx: &Sender<TrainingEvent>,
    ) -> Result<ReqResolution> {
        let id = self.id;

        let should_continue = match req {
            WorkerRequest::Disconnect => {
                info!("disconnecting worker {id}");
                let event = TrainingEvent::Disconnect { worker_id: id };

                if let Err(e) = self.worker_handle.disconnect().await {
                    let _ = event_tx.send(event).await;
                    let details = format!("failed to disconnect worker {id}: {e}");
                    return Err(OrchErr::WorkerError { id, details });
                }

                let _ = event_tx.send(event).await;
                ReqResolution::Halt
            }
            WorkerRequest::Stop => {
                if !self.stopping {
                    info!("stopping worker {id}");

                    if let Err(e) = self.worker_handle.stop().await {
                        let details = format!("failed to stop worker {id}: {e}");
                        return Err(OrchErr::WorkerError { id, details });
                    }

                    self.stopping = true;
                }

                ReqResolution::Continue
            }
            WorkerRequest::PullParams => {
                info!("pulling parmaeters from worker {id}");

                match self.worker_handle.pull_params().await {
                    Ok(params) => {
                        let event = TrainingEvent::Params(params.to_vec());
                        let _ = event_tx.send(event).await;
                    }
                    Err(e) => {
                        let details = format!("failed to pull params from worker {id}: {e}");
                        return Err(OrchErr::WorkerError { id, details });
                    }
                }

                ReqResolution::Continue
            }
            WorkerRequest::Switch {
                server_addrs,
                server_sizes,
                server_ordering,
            } => {
                if let Err(e) = self
                    .worker_handle
                    .switch(server_addrs, server_sizes, server_ordering)
                    .await
                {
                    let details = format!("failed to switch worker {id}: {e}");
                    return Err(OrchErr::WorkerError { id, details });
                }

                ReqResolution::Continue
            }
            WorkerRequest::Upgrade { spec, ranges } => ReqResolution::Upgrade { spec, ranges },
        };

        Ok(should_continue)
    }

    /// Handles the worker's events for the orchestrator.
    ///
    /// # Args
    /// * `id` - The worker's id.
    /// * `event` - The worker event to handle.
    ///
    /// # Returns
    /// An `EventResolution` or an orch error if the received event is invalid.
    fn handle_event(id: usize, event: WorkerEvent<'_>) -> Result<EventResolution> {
        match event {
            WorkerEvent::Loss(losses) => {
                debug!("worker {id} reported {} losses", losses.len());

                let training_event = TrainingEvent::PublishedLosses {
                    worker_id: id,
                    losses,
                };

                Ok(EventResolution::NotifyOrch(training_event))
            }
            WorkerEvent::Done => {
                info!("worker {id} done");
                let training_event = TrainingEvent::WorkerDone(id);
                Ok(EventResolution::NotifyOrch(training_event))
            }
            WorkerEvent::Disconnect => {
                info!("worker {id} disconnected");
                Ok(EventResolution::Exit)
            }
            _ => {
                warn!("worker {id}: unexpected event {event:?}");
                let details = format!("invalid event from worker {id}: {event:?}");
                Err(OrchErr::WorkerError { id, details })
            }
        }
    }
}
