use std::io;

use comms::{
    OnoReceiver, OnoSender,
    msg::{Command, Msg, Payload},
};
use tokio::{
    io::{AsyncRead, AsyncWrite},
    task::JoinSet,
};

use crate::{
    optimization::Optimizer,
    service::Server,
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

    /// Creates an error for when an unexpected message kind is received.
    ///
    /// # Arguments
    /// * `msg` - The received message.
    ///
    /// # Returns
    /// An error.
    fn unexpected_message_kind<U>(msg: Msg) -> io::Result<U> {
        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Received an unexpected message kind, got: {msg:?}"),
        ))
    }
}

impl<O: Optimizer + Send, T: Trainer> ParameterServer<O, T> {
    /// Starts the training process with the spawned workers.
    ///
    /// # Returns
    /// The trained weights of the model.
    pub async fn train(&mut self) -> io::Result<Vec<f32>> {
        while let Some(ret) = self.tasks.join_next().await {
            ret??;
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
        let handle = self.handle.clone();
        let trainer = self.trainer.clone();
        let mut buf = vec![0.; handle.len()];

        let task = async move {
            // SAFETY: This buffer is the same size as the
            //         amount of parameters in the storage.
            handle.pull_weights(&mut buf).await.unwrap();

            loop {
                let msg = Msg::Data(Payload::Weights(&mut buf));
                tx.send(&msg).await?;

                match rx.recv().await? {
                    Msg::Data(Payload::Gradient(grad)) => {
                        trainer.step(&handle, grad, &mut buf).await;
                    }
                    Msg::Control(Command::Disconnect) => break,
                    _ => return Self::unexpected_message_kind(msg),
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
    async fn train(&mut self) -> io::Result<Vec<f32>> {
        self.train().await
    }

    fn spawn(&mut self, rx: OnoReceiver<R>, tx: OnoSender<W>) {
        self.spawn(rx, tx)
    }
}
