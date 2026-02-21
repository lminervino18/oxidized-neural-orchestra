use std::io;

use comms::{
    OnoReceiver, OnoSender,
    msg::{Command, Detail, Msg, Payload},
};
use log::{debug, error, info, warn};
use tokio::{
    io::{AsyncRead, AsyncWrite},
    task::JoinSet,
};

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
    /// # Arguments
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
    /// # Arguments
    /// * `rx` - The receiving end of the communication.
    /// * `tx` - The sending end of the communication.
    pub fn spawn<R, W>(&mut self, mut rx: OnoReceiver<R>, mut tx: OnoSender<W>)
    where
        R: AsyncRead + Unpin + Send + 'static,
        W: AsyncWrite + Unpin + Send + 'static,
    {
        let id = self.tasks.len() + 1;
        let handle = self.handle.clone();
        let synchronizer = self.synchronizer.clone();
        let mut msg_buf = vec![0; 1028];

        let task = async move {
            let nparams = handle.len();
            let mut params = vec![0.; nparams];

            // SAFETY: This buffer is the same size as the
            //         amount of parameters in the storage.
            handle.pull_params(&mut params).await.unwrap();

            debug!(worker_id = id; "sending parameters");
            let msg = Msg::Data(Payload::Params(&mut params));
            tx.send(&msg).await?;

            loop {
                debug!(worker_id = id; "waiting to receive a message");
                match rx.recv_into(&mut msg_buf).await? {
                    Msg::Data(Payload::Grad(grad)) if nparams == grad.len() => {
                        debug!(worker_id = id; "received gradient, applying step");

                        // SAFETY: We checked that the gradient is the same
                        //         size as the buffer and the storage.
                        synchronizer.step(&handle, grad, &mut params).await.unwrap();

                        debug!(worker_id = id; "sending parameters");
                        let msg = Msg::Data(Payload::Params(&mut params));
                        tx.send(&msg).await?;
                    }
                    Msg::Control(Command::Disconnect) => {
                        info!(worker_id = id; "gracefully disconnecting worker");
                        let msg = Msg::Control(Command::Disconnect);
                        tx.send(&msg).await?;
                        break;
                    }
                    Msg::Data(Payload::Grad(grad)) => {
                        let ngrad = grad.len();
                        warn!(worker_id = id; "gradient size mismatch, expected {nparams}, got {ngrad}");

                        let msg = Msg::Err(Detail::BufferSizeMismatch {
                            expected: nparams,
                            got: ngrad,
                        });

                        tx.send(&msg).await?;
                    }
                    msg => {
                        error!(worker_id = id; "received an invalid message {msg:?}");

                        let msg = Msg::Err(Detail::Fatal("invalid message".into()));
                        tx.send(&msg).await?;

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
impl<R, W, PS, Sy> Server<R, W> for ParameterServer<PS, Sy>
where
    R: AsyncRead + Unpin + Send + 'static,
    W: AsyncWrite + Unpin + Send + 'static,
    PS: Store + Send + Sync + 'static,
    Sy: Synchronizer + 'static,
{
    async fn run(&mut self) -> io::Result<Vec<f32>> {
        self.run().await
    }

    fn spawn(&mut self, rx: OnoReceiver<R>, tx: OnoSender<W>) {
        self.spawn(rx, tx)
    }
}
