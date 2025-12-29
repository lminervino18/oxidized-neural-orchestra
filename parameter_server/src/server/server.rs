use std::{borrow::Cow, io};

use comms::{
    OnoReceiver, OnoSender,
    msg::{Msg, Payload},
};
use tokio::{
    io::{AsyncRead, AsyncWrite},
    task::JoinSet,
};

use crate::execution::Executor;

/// The central server structure, it manages task spawn/join and io with external worker nodes.
pub struct ParameterServer<E: Executor> {
    tasks: JoinSet<io::Result<()>>,
    params: usize,
    epochs: usize,
    executor: E,
}

impl<E: Executor> ParameterServer<E> {
    /// Creates a new `ParameterServer`.
    ///
    /// # Arguments
    /// * `params` - The amount of parameters to hold.
    /// * `epochs` - The amount of epochs of training to run.
    /// * `executor` - The executor to use in training.
    pub fn new(params: usize, epochs: usize, executor: E) -> Self {
        Self {
            tasks: JoinSet::new(),
            params,
            epochs,
            executor,
        }
    }

    /// Starts the actual training by executing the inner tasks.
    pub async fn run(self) -> Vec<io::Result<()>> {
        self.tasks.join_all().await
    }
}

impl<E: Executor + Send + Sync + 'static> ParameterServer<E> {
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
        let params = self.params;
        let executor = self.executor.clone();
        let mut buf = vec![0.; params];

        let task = async move {
            loop {
                let msg: Msg = rx.recv().await?;

                let grad = match msg {
                    Msg::Data(Payload::Gradient(grad)) => grad,
                    _ => unreachable!(),
                };

                executor.step(&grad, &mut buf).await;
                let msg = Msg::Data(Payload::Weights(Cow::Borrowed(&buf)));
                tx.send(&msg).await?;
            }
        };

        self.tasks.spawn(task);
    }
}
