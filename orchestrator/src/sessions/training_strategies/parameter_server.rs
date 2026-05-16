use comms::{NetRtp, ParamServerCluster, TransportLayer, WorkerEvent, WorkerHandle};
use log::{debug, error, info, warn};
use tokio::sync::mpsc::{Receiver, Sender};

use super::WorkerRequest;
use crate::{OrchErr, Result, TrainingEvent, sessions::TrainingStrategy};

/// The strategy used when the training is using the Parameter Server algorithm.
pub struct ParameterServerTrainingStrategy<T>
where
    T: TransportLayer,
{
    workers: Vec<WorkerHandle<T>>,
    cluster: ParamServerCluster<T>,
}

impl<T> ParameterServerTrainingStrategy<T>
where
    T: TransportLayer,
{
    /// Crates a new `ParameterServerTrainingStrategy`.
    ///
    /// # Args
    /// * `cluster` - The server cluster communicating handle.
    ///
    /// # Returns
    /// A new `ParamServertrainingStrategy` instance.
    pub fn new(cluster: ParamServerCluster<T>) -> Self {
        Self {
            workers: Vec::new(),
            cluster,
        }
    }
}

impl<T> TrainingStrategy for ParameterServerTrainingStrategy<T>
where
    T: TransportLayer,
{
    async fn spawn(
        id: usize,
        mut worker_handle: WorkerHandle<NetRtp>,
        mut rx: Receiver<WorkerRequest>,
        tx: Sender<TrainingEvent>,
    ) {
        loop {
            tokio::select! {
                req = rx.recv() => {
                    if let Some(req) = req {
                        if let Err(e) = handle_request(id, &mut worker_handle, req).await {
                            let _ = tx.send(TrainingEvent::Error(e)).await;
                        }
                    }
                },
                event = worker_handle.recv_event() => match event {
                    Ok(event) => match handle_event::<T>(id, event) {
                        Ok(EventResolution::Exit) => break,
                        Ok(EventResolution::NotifyOrch(event)) => {
                            let _ = tx.send(event).await;
                        }
                        Err(e) => {
                            let _ = tx.send(TrainingEvent::Error(e)).await;
                            return;
                        }
                    }
                    Err(e) => {
                        error!("worker {id} error: {e}");
                        let err = OrchErr::WorkerError { id, details: e.to_string() };
                        let _ = tx.send(TrainingEvent::Error(err)).await;
                        return;
                    }
                }
            }
        }
    }
}

/// Handles the orchestrator's requests for a worker.
///
/// # Args
/// * `id` - The id of the worker being requested.
/// * `worker_handle` - The worker's communication handle.
/// * `req` - The request that was made.
///
/// # Returns
/// An orch error if occurred.
async fn handle_request<T>(
    id: usize,
    worker_handle: &mut WorkerHandle<T>,
    req: WorkerRequest,
) -> Result<()>
where
    T: TransportLayer,
{
    match req {
        WorkerRequest::Disconnect => {
            if let Err(e) = worker_handle.disconnect().await {
                let event = format!("failed to disconnect worker {id}: {e}");
                return Err(OrchErr::WorkerError { id, details: event });
            }
        }
        WorkerRequest::Stop => {
            if let Err(e) = worker_handle.stop().await {
                let event = format!("failed to stop worker {id}: {e}");
                return Err(OrchErr::WorkerError { id, details: event });
            }
        }
        _ => return Err(OrchErr::InvalidRequest(req)),
    }

    Ok(())
}

/// The return value of the `handle_event` function.
enum EventResolution {
    NotifyOrch(TrainingEvent),
    Exit,
}

/// Handles the worker's events for the orchestrator.
///
/// # Args
/// * `id` - The worker's id.
/// * `event` - The worker event to handle.
///
/// # Returns
/// An `EventResolution` or an orch error if the received event is invalid.
fn handle_event<T>(id: usize, event: WorkerEvent<'_>) -> Result<EventResolution>
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
