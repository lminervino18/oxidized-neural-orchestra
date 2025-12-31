use std::{borrow::Cow, io};

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
            for _ in 0..epochs {
                let msg: Msg = rx.recv().await?;

                let Msg::Data(Payload::Gradient(grad)) = msg else {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("Received an invalid message kind {msg:?}"),
                    ));
                };

                trainer.step(&grad, &mut buf).await;

                let msg = Msg::Data(Payload::Weights(Cow::Borrowed(&buf)));
                tx.send(&msg).await?;
            }

            Ok(())
        };

        self.tasks.spawn(task);
    }
}
