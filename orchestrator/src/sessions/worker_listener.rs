use comms::{TransportLayer, WorkerEvent, WorkerHandle};
use log::{debug, error, info, warn};
use tokio::sync::mpsc::{Receiver, Sender};

use super::{TrainingEvent, WorkerRequest};
use crate::{OrchErr, Result};

/// The return type of the `handle_event` method.
enum EventResolution {
    NotifyOrch(TrainingEvent),
    Exit,
}

/// The worker handle manager.
pub struct WorkerListener<T>
where
    T: TransportLayer,
{
    id: usize,
    worker_handle: WorkerHandle<T>,
    stopping: bool,
}

impl<T> WorkerListener<T>
where
    T: TransportLayer,
{
    /// Creates a new `WorkerListener`.
    ///
    /// # Args
    /// * `id` - The id of the worker being requested.
    /// * `worker_handle` - The worker's communication handle.
    ///
    /// # Returns
    /// A new `WorkerListener` instance.
    pub fn new(id: usize, worker_handle: WorkerHandle<T>) -> Self {
        Self {
            id,
            worker_handle,
            stopping: false,
        }
    }

    /// Starts the worker listener.
    ///
    /// # Args
    /// * `rx` - The request receiver.
    /// * `tx` - The training events sender.
    ///
    /// # Returns
    /// An orch error if occurred.
    pub async fn listen(mut self, mut rx: Receiver<WorkerRequest>, tx: Sender<TrainingEvent>) {
        let id = self.id;

        loop {
            tokio::select! {
                req = rx.recv() => {
                    if let Some(req) = req
                        && let Err(e) = self.handle_request(req).await
                    {
                        let _ = tx.send(TrainingEvent::Error(e)).await;
                    }
                },
                event = self.worker_handle.recv_event() => match event {
                    Ok(event) => match Self::handle_event(id, event) {
                        Ok(EventResolution::Exit) => break,
                        Ok(EventResolution::NotifyOrch(event)) => {
                            let _ = tx.send(event).await;
                        }
                        Err(e) => {
                            let _ = tx.send(TrainingEvent::Error(e)).await;
                            break;
                        }
                    }
                    Err(e) => {
                        error!("worker {id} error: {e}");
                        let err = OrchErr::WorkerError { id, details: e.to_string() };
                        let _ = tx.send(TrainingEvent::Error(err)).await;
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
    ///
    /// # Returns
    /// An orch error if occurred.
    async fn handle_request(&mut self, req: WorkerRequest) -> Result<()>
    where
        T: TransportLayer,
    {
        let id = self.id;

        match req {
            WorkerRequest::Disconnect => {
                if let Err(e) = self.worker_handle.disconnect().await {
                    info!("disconnecting worker {id}");
                    let details = format!("failed to disconnect worker {id}: {e}");
                    return Err(OrchErr::WorkerError { id, details });
                }
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
            }
            WorkerRequest::PullParams => {
                info!("pulling parmaeters from worker {id}");

                if let Err(e) = self.worker_handle.pull_params().await {
                    let details = format!("failed to pull params from worker {id}: {e}");
                    return Err(OrchErr::WorkerError { id, details });
                }
            }
        }

        Ok(())
    }

    /// Handles the worker's events for the orchestrator.
    ///
    /// # Args
    /// * `id` - The worker's id.
    /// * `event` - The worker event to handle.
    ///
    /// # Returns
    /// An `EventResolution` or an orch error if the received event is invalid.
    fn handle_event(id: usize, event: WorkerEvent<'_>) -> Result<EventResolution>
    where
        T: TransportLayer,
    {
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
