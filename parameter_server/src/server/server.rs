use std::io;

use comms::{
    OnoReceiver, OnoSender,
    msg::{Msg, Payload},
};
use tokio::{
    io::{AsyncRead, AsyncWrite},
    task::JoinSet,
};

use crate::training::Trainer;

/// The central server structure, it handles task management and io between workers.
pub struct ParameterServer<T: Trainer> {
    tasks: JoinSet<io::Result<()>>,
    params: usize,
    epochs: usize,
    trainer: T,
}

impl<T: Trainer> ParameterServer<T> {
    /// Creates a new `ParameterServer`.
    ///
    /// # Arguments
    /// * `params` - The amount of parameters to hold.
    /// * `epochs` - The amount of epochs of training to run.
    /// * `trainer` - The trainer to use.
    pub fn new(params: usize, epochs: usize, trainer: T) -> Self {
        Self {
            tasks: JoinSet::new(),
            params,
            epochs,
            trainer,
        }
    }

    /// Starts the actual training by executing the inner tasks.
    pub async fn run(self) -> Vec<io::Result<()>> {
        self.tasks.join_all().await
    }

    /// Creates an error for when an unexpected message kind is received.
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

impl<T: Trainer + Send + Sync + 'static> ParameterServer<T> {
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
        let Self { epochs, params, .. } = *self;
        let trainer = self.trainer.clone();
        let mut buf = vec![0.; params];

        let task = async move {
            trainer.pull_weights(&mut buf).await;

            for _ in 0..epochs {
                let msg = Msg::Data(Payload::Weights(&buf));
                tx.send(&msg).await?;

                let msg: Msg = rx.recv().await?;
                let Msg::Data(Payload::Gradient(grad)) = msg else {
                    return Self::unexpected_message_kind(msg);
                };

                trainer.step(&grad, &mut buf).await;
            }

            Ok(())
        };

        self.tasks.spawn(task);
    }
}
