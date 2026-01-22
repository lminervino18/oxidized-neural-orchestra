use std::io;

use comms::{
    msg::{Msg, Payload},
    OnoReceiver, OnoSender,
};
use log::{debug, error, info, warn};
use machine_learning::TrainStrategy;
use tokio::io::{AsyncRead, AsyncWrite};

use crate::config::WorkerConfig;

/// Infrastructure worker runtime.
pub struct Worker<S: TrainStrategy> {
    cfg: WorkerConfig,
    strategy: S,
    grads_buf: Vec<f32>,
}

impl<S: TrainStrategy> Worker<S> {
    /// Creates a new `Worker`.
    ///
    /// # Args
    /// * `cfg` - Immutable worker configuration.
    /// * `strategy` - Training strategy used to compute gradients.
    ///
    /// # Returns
    /// A new `Worker` instance.
    ///
    /// # Panics
    /// Never panics.
    pub fn new(cfg: WorkerConfig, strategy: S) -> Self {
        Self {
            cfg,
            strategy,
            grads_buf: Vec::new(),
        }
    }

    /// Runs the worker loop for the configured number of steps.
    ///
    /// The parameter count is inferred from the first `Weights` message received.
    /// Subsequent `Weights` messages must match that inferred size.
    ///
    /// # Args
    /// * `rx` - Receiving end of the communication channel.
    /// * `tx` - Sending end of the communication channel.
    ///
    /// # Returns
    /// Returns `Ok(())` if all steps complete successfully.
    ///
    /// # Errors
    /// Returns `io::Error` if receiving/sending fails, an unexpected message is received,
    /// or a `Weights` payload length differs from the first observed weight vector.
    pub async fn run<R, W>(mut self, mut rx: OnoReceiver<R>, mut tx: OnoSender<W>) -> io::Result<()>
    where
        R: AsyncRead + Unpin + Send,
        W: AsyncWrite + Unpin + Send,
    {
        let worker_id = self.cfg.worker_id();

        info!(worker_id = worker_id; "worker starting");

        for step in 0..self.cfg.steps() {
            debug!(worker_id = worker_id, step = step; "waiting for weights");
            let msg: Msg = rx.recv().await?;

            let weights = match msg {
                Msg::Data(Payload::Weights(w)) => w,
                other => {
                    error!(worker_id = worker_id, step = step; "unexpected message");
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("unexpected message: {other:?}"),
                    ));
                }
            };

            if self.grads_buf.is_empty() {
                self.grads_buf.resize(weights.len(), 0.0);
            } else if weights.len() != self.grads_buf.len() {
                let got = weights.len();
                let expected = self.grads_buf.len();

                error!(
                    worker_id = worker_id,
                    step = step,
                    got = got,
                    expected = expected;
                    "weights length mismatch"
                );
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("weights length mismatch: got {got}, expected {expected}"),
                ));
            }

            self.grads_buf.fill(0.0);

            debug!(worker_id = worker_id, step = step; "computing gradients");
            let res =
                tokio::task::block_in_place(|| self.strategy.step(weights, &mut self.grads_buf));
            if let Err(e) = res {
                warn!(worker_id = worker_id, step = step; "train strategy error: {e}");
                return Err(io::Error::new(io::ErrorKind::InvalidData, e.to_string()));
            }

            debug!(worker_id = worker_id, step = step; "sending gradient");
            let out = Msg::Data(Payload::Gradient(&self.grads_buf));
            tx.send(&out).await?;
        }

        info!(worker_id = worker_id; "worker finished");
        Ok(())
    }
}
