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
    optimization::Optimizer,
    storage::{ParameterHandle, ParameterStore},
    training::Trainer,
};

/// The central server structure, it handles task management and io between workers.
pub struct ParameterServer<O: Optimizer, T: Trainer> {
    tasks: JoinSet<io::Result<()>>,
    handle: ParameterHandle<O>,
    trainer: T,
}

impl<O: Optimizer, T: Trainer> ParameterServer<O, T> {
    /// Creates a new `ParameterServer`.
    ///
    /// # Arguments
    /// * `store` - The underlying parameter store to use.
    /// * `trainer` - The trainer to use.
    pub fn new(store: ParameterStore<O>, trainer: T) -> Self {
        Self {
            tasks: JoinSet::new(),
            handle: ParameterHandle::new(store),
            trainer,
        }
    }
}

impl<O: Optimizer + Send, T: Trainer> ParameterServer<O, T> {
    /// Starts the training process with the spawned workers.
    ///
    /// # Returns
    /// The trained weights of the model.
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

        // SAFETY: This weight vector is the same size as
        //         the amount of parameters in the storage.
        let params = self.handle.len();
        let mut weights = vec![0.; params];
        self.handle.pull_weights(&mut weights).await.unwrap();

        Ok(weights)
    }
}

impl<O: Optimizer + Send + 'static, T: Trainer + 'static> ParameterServer<O, T> {
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
        let trainer = self.trainer.clone();

        let task = async move {
            let params = handle.len();
            let mut buf = vec![0.; params];

            // SAFETY: This buffer is the same size as the
            //         amount of parameters in the storage.
            handle.pull_weights(&mut buf).await.unwrap();

            loop {
                debug!(worker_id = id; "sending weights");
                let msg = Msg::Data(Payload::Weights(&mut buf));
                tx.send(&msg).await?;

                debug!(worker_id = id; "waiting to receive a message");
                match rx.recv().await? {
                    Msg::Data(Payload::Gradient(grad)) if params == grad.len() => {
                        debug!(worker_id = id; "received gradient, applying step");

                        // SAFETY: We checked that the gradient is the same
                        //         size as the buffer and the storage.
                        trainer.step(&handle, grad, &mut buf).await.unwrap();
                    }
                    Msg::Control(Command::Disconnect) => {
                        info!(worker_id = id; "gracefully disconnecting worker");
                        break;
                    }
                    Msg::Data(Payload::Gradient(grad)) => {
                        warn!(worker_id = id; "gradient size mismatch, expected {params}, got {}", grad.len());

                        let msg = Msg::Err(Detail::GradSizeMismatch {
                            expected: params,
                            got: grad.len(),
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
impl<R, W, O, T> Server<R, W> for ParameterServer<O, T>
where
    R: AsyncRead + Unpin + Send + 'static,
    W: AsyncWrite + Unpin + Send + 'static,
    O: Optimizer + Send + 'static,
    T: Trainer + 'static,
{
    async fn run(&mut self) -> io::Result<Vec<f32>> {
        self.run().await
    }

    fn spawn(&mut self, rx: OnoReceiver<R>, tx: OnoSender<W>) {
        self.spawn(rx, tx)
    }
}
