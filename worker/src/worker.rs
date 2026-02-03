use comms::{
    msg::{Msg, Payload},
    OnoReceiver, OnoSender,
};
use log::{debug, error, info, warn};
use tokio::io::{AsyncRead, AsyncWrite};

use crate::error::WorkerError;
use machine_learning::training::Trainer;

/// Infrastructure worker runtime.
pub struct Worker {
    worker_id: usize,
    trainer: Box<dyn Trainer>,
}

impl Worker {
    /// Creates a new `Worker`.
    ///
    /// # Args
    /// * `worker_id` - Worker identifier used for observability.
    /// * `cfg` - Immutable execution bounds.
    /// * `num_params` - Expected parameter count for `weights` and `grads`.
    /// * `optimizer` - Worker-local optimizer instance.
    /// * `offline_steps` - Local update iterations per received snapshot.
    ///
    /// # Returns
    /// A new `Worker` instance.
    pub fn new(worker_id: usize, trainer: Box<dyn Trainer>) -> Self {
        Self { worker_id, trainer }
    }

    /// Runs the worker loop for the configured number of steps.
    ///
    /// # Args
    /// * `rx` - Receiving end of the communication channel.
    /// * `tx` - Sending end of the communication channel.
    ///
    /// # Returns
    /// Returns `Ok(())` if all steps complete successfully.
    ///
    /// # Errors
    /// Returns `WorkerError` on I/O failures, protocol violations, or ML failures.
    pub async fn run<R, W>(
        self,
        mut rx: OnoReceiver<R>,
        mut tx: OnoSender<W>,
    ) -> Result<(), WorkerError>
    where
        R: AsyncRead + Unpin + Send,
        W: AsyncWrite + Unpin + Send,
    {
        let worker_id = self.worker_id;
        let mut trainer = self.trainer;

        info!(worker_id = worker_id; "worker starting");

        match rx.recv().await.unwrap() {
            Msg::Data(Payload::Weights(params)) => {
                let grad = trainer.train(params);
                let msg = Msg::Data(Payload::Gradient(grad));
                tx.send(&msg).await.unwrap();
            }
            _ => todo!(),
        }

        info!(worker_id = worker_id; "worker finished");
        Ok(())
    }
}

fn msg_kind(msg: &Msg<'_>) -> &'static str {
    match msg {
        Msg::Control(_) => "control",
        Msg::Err(_) => "err",
        Msg::Data(Payload::Gradient(_)) => "data/gradient",
        Msg::Data(Payload::Weights(_)) => "data/weights",
    }
}
