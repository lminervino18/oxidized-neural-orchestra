use std::io;

use comms::{TransportLayer, WorkerEvent, WorkerHandle};
use log::{debug, error, info, warn};
use tokio::task::JoinSet;

use super::Server;
use crate::{
    storage::{Store, StoreHandle},
    synchronization::Synchronizer,
};

/// The central server structure, it handles task management and io between workers.
pub struct ParameterServer<PS: Store, Sy: Synchronizer> {
    tasks: JoinSet<io::Result<()>>,
    handle: StoreHandle<PS>,
    synchronizer: Sy,
}

impl<PS: Store, Sy: Synchronizer> ParameterServer<PS, Sy> {
    /// Creates a new `ParameterServer`.
    ///
    /// # Args
    /// * `handle` - The underlying parameter store to use behind a handle.
    /// * `synchronizer` - The synchronizer to use.
    ///
    /// # Returns
    /// A new `ParameterServer` instance.
    pub fn new(handle: StoreHandle<PS>, synchronizer: Sy) -> Self {
        Self {
            tasks: JoinSet::new(),
            handle,
            synchronizer,
        }
    }
}

impl<PS: Store, Sy: Synchronizer> ParameterServer<PS, Sy> {
    /// Starts the training process with the spawned workers.
    ///
    /// # Returns
    /// The trained parameters of the model.
    pub async fn run(&mut self) -> io::Result<Vec<f32>> {
        while let Some(ret) = self.tasks.join_next().await {
            match ret {
                Ok(Err(e)) => {
                    error!("worker task failed with error: {e}");
                    return Err(e);
                }
                Err(e) => {
                    error!("task panicked or was cancelled: {e}");
                    return Err(io::Error::other(e));
                }
                _ => {}
            }
        }

        // SAFETY: This parameter vector is the same size as
        //         the amount of parameters in the storage.
        let nparams = self.handle.len();
        let mut params = vec![0.; nparams];
        self.handle.pull_params(&mut params).await.unwrap();
        Ok(params)
    }
}

impl<PS: Store + Send + Sync + 'static, Sy: Synchronizer + 'static> ParameterServer<PS, Sy> {
    /// Binds a new worker to this server and spawns it's own training task.
    ///
    /// # Args
    /// * `worker_handle` - The handle for a worker connection.
    pub fn spawn<T>(&mut self, mut worker_handle: WorkerHandle<T>)
    where
        T: TransportLayer + Send + 'static,
    {
        let id = self.tasks.len() + 1;
        let handle = self.handle.clone();
        let synchronizer = self.synchronizer.clone();

        let task = async move {
            let nparams = handle.len();
            let mut params = vec![0.0; nparams];

            // SAFETY: This buffer is the same size as the
            //         amount of parameters in the storage.
            handle.pull_params(&mut params).await.unwrap();

            loop {
                debug!(worker_id = id; "waiting to receive a message");

                match worker_handle.recv_event().await? {
                    WorkerEvent::RequestParams => {
                        debug!(worker_id = id; "sending parameters");
                        worker_handle.push_params(&mut params).await?;
                    }
                    WorkerEvent::Grad(grad) if nparams == grad.len() => {
                        debug!(worker_id = id; "received gradient, applying step");

                        // SAFETY: We checked that the gradient is the same
                        //         size as the buffer and the storage.
                        synchronizer
                            .step(&handle, grad, &mut params)
                            .await
                            .map_err(io::Error::other)?;
                    }
                    WorkerEvent::Disconnect => {
                        info!(worker_id = id; "gracefully disconnecting worker");
                        // Release any peer tasks blocked inside a barrier step so they
                        // are not left waiting for a contribution that will never arrive.
                        synchronizer.drain();
                        break;
                    }
                    WorkerEvent::Grad(grad) => {
                        // Acá eventualmente habría que ver como se redimensiona el servidor a partir
                        // de que haya llegado otro tamaño de gradiente.
                        //
                        // ¿Cómo agregamos o quitamos valores al store y demás?
                        warn!(worker_id = id; "gradient size mismatch, expected {nparams}, got {}", grad.len());
                    }
                    event => {
                        error!(worker_id = id; "received an invalid event {event:?}");
                        return Err(io::Error::other("invalid message"));
                    }
                }
            }

            Ok(())
        };

        self.tasks.spawn(task);
    }
}

#[async_trait::async_trait]
impl<T, PS, Sy> Server<T> for ParameterServer<PS, Sy>
where
    T: TransportLayer + Send + 'static,
    PS: Store + Send + Sync + 'static,
    Sy: Synchronizer + Send + 'static,
{
    async fn run(&mut self) -> io::Result<Vec<f32>> {
        self.run().await
    }

    fn spawn(&mut self, worker_handle: WorkerHandle<T>) {
        self.spawn(worker_handle)
    }
}
