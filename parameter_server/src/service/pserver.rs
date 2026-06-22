use std::io;

use comms::{OrchEvent, OrchHandle, TransportLayer, WorkerEvent, WorkerHandle};
use log::{debug, error, info, warn};
use tokio::{
    io::{AsyncRead, AsyncWrite},
    task::JoinSet,
};

use super::Server;
use crate::{storage::Store, synchronization::Synchronizer};

/// The central server structure, it handles task management and io between workers.
pub struct ParameterServer<R, W, T, PS, Sy>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
    T: TransportLayer<R, W>,
    PS: Store,
    Sy: Synchronizer,
{
    tasks: JoinSet<io::Result<()>>,
    store: PS,
    synchronizer: Sy,
    orch_handle: OrchHandle<R, W, T>,
}

impl<R, W, T, PS, Sy> ParameterServer<R, W, T, PS, Sy>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
    T: TransportLayer<R, W>,
    PS: Store,
    Sy: Synchronizer,
{
    /// Creates a new `ParameterServer`.
    ///
    /// # Args
    /// * `store` - The underlying parameter store to use.
    /// * `synchronizer` - The synchronizer to use.
    ///
    /// # Returns
    /// A new `ParameterServer` instance.
    pub fn new(store: PS, synchronizer: Sy, orch_handle: OrchHandle<R, W, T>) -> Self {
        Self {
            tasks: JoinSet::new(),
            store,
            synchronizer,
            orch_handle,
        }
    }
}

impl<R, W, T, PS, Sy> ParameterServer<R, W, T, PS, Sy>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
    T: TransportLayer<R, W>,
    PS: Store,
    Sy: Synchronizer,
{
    /// Starts the training process with the spawned workers.
    ///
    /// # Returns
    /// The trained parameters of the model.
    pub async fn run(&mut self) -> io::Result<()> {
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

        let nparams = self.store.len();
        let mut params = vec![0.0; nparams];

        // SAFETY: The parameter vector is the same size as
        //         the amount of parameters in the storage.
        self.store.pull_params(&mut params).unwrap();

        loop {
            let event = self.orch_handle.recv_event().await?;

            match event {
                OrchEvent::Disconnect => break,
                OrchEvent::RequestParams => self.orch_handle.push_params(&mut params).await?,
                event => warn!("Unexpected OrchEvent: {event:?}"),
            }
        }

        Ok(())
    }
}

impl<R, W, T, PS, Sy> ParameterServer<R, W, T, PS, Sy>
where
    R: AsyncRead + Unpin + Send + 'static,
    W: AsyncWrite + Unpin + Send + 'static,
    T: TransportLayer<R, W> + 'static,
    PS: Store + Send + Sync + 'static,
    Sy: Synchronizer + 'static,
{
    /// Binds a new worker to this server and spawns it's own training task.
    ///
    /// # Args
    /// * `worker_handle` - The handle for a worker connection.
    pub fn spawn(&mut self, mut worker_handle: WorkerHandle<R, W, T>) {
        let id = self.tasks.len() + 1;
        let store = self.store.clone();
        let synchronizer = self.synchronizer.clone();

        let task = async move {
            let nparams = store.len();
            let mut params = vec![0.0; nparams];

            // SAFETY: This buffer is the same size as the
            //         amount of parameters in the storage.
            store.pull_params(&mut params).unwrap();
            worker_handle.push_params(&mut params).await?;

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
                            .step(&store, grad, &mut params)
                            .await
                            .map_err(io::Error::other)?;

                        worker_handle.push_params(&mut params).await?;
                    }
                    WorkerEvent::Disconnect => {
                        info!(worker_id = id; "gracefully disconnecting worker");
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
impl<R, W, T, PS, Sy> Server<R, W, T> for ParameterServer<R, W, T, PS, Sy>
where
    R: AsyncRead + Unpin + Send + 'static,
    W: AsyncWrite + Unpin + Send + 'static,
    PS: Store + Send + Sync + 'static,
    Sy: Synchronizer + Send + 'static,
    T: TransportLayer<R, W> + Send + 'static,
{
    async fn run(&mut self) -> io::Result<()> {
        self.run().await
    }

    fn spawn(&mut self, worker_handle: WorkerHandle<R, W, T>) {
        self.spawn(worker_handle)
    }
}
