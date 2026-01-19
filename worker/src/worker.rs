use std::{io, num::NonZeroUsize};

use comms::{
    msg::{Msg, Payload},
    OnoReceiver, OnoSender,
};
use log::{debug, error, info, warn};
use ml_core::TrainStrategy;
use tokio::io::{AsyncRead, AsyncWrite};

use crate::config::WorkerConfig;

/// Infrastructure worker runtime.
///
/// The worker executes a fixed receive → compute → send loop and delegates all
/// training logic to an injected `TrainStrategy`. This type does not own model
/// weights or datasets.
pub struct Worker<S: TrainStrategy> {
    cfg: WorkerConfig,
    strategy: S,
    grads_buf: Vec<f32>,
}

impl<S: TrainStrategy> Worker<S> {
    /// Creates a new `Worker`.
    ///
    /// # Args
    /// * `cfg` - Immutable worker configuration (identity and execution bounds).
    /// * `num_params` - Number of model parameters expected in both `weights` and `grads`.
    /// * `strategy` - Training strategy responsible for computing local gradients.
    ///
    /// # Returns
    /// A fully initialized `Worker` instance with a persistent gradient buffer sized
    /// according to `num_params`.
    ///
    /// # Panics
    /// Never panics. Structural invariants are enforced via types.
    pub fn new(cfg: WorkerConfig, num_params: NonZeroUsize, strategy: S) -> Self {
        let n = num_params.get();
        Self {
            cfg,
            strategy,
            grads_buf: vec![0.0; n],
        }
    }

    /// Runs the worker loop for the configured number of steps.
    ///
    /// The worker expects to receive `Msg::Data(Payload::Weights)` messages and will
    /// respond with `Msg::Data(Payload::Gradient)` messages.
    ///
    /// # Args
    /// * `rx` - Receiving end of the communication channel.
    /// * `tx` - Sending end of the communication channel.
    ///
    /// # Returns
    /// Returns `Ok(())` if all steps complete successfully. Returns `io::Error` if
    /// communication fails, an unexpected message is received, or input validation fails.
    ///
    /// # Errors
    /// - Returns `InvalidData` if an unexpected message kind is received.
    /// - Returns `InvalidData` if the received weight slice length does not match the
    ///   worker's gradient buffer length.
    /// - Returns `InvalidData` if the training strategy reports an invalid input.
    ///
    /// # Panics
    /// Never panics.
    pub async fn run<R, W>(
        mut self,
        mut rx: OnoReceiver<R>,
        mut tx: OnoSender<W>,
    ) -> io::Result<()>
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

            if weights.len() != self.grads_buf.len() {
                error!(
                    worker_id = worker_id,
                    step = step,
                    got = weights.len(),
                    expected = self.grads_buf.len();
                    "weights length mismatch"
                );
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "weights length mismatch: got {}, expected {}",
                        weights.len(),
                        self.grads_buf.len()
                    ),
                ));
            }

            debug!(
                worker_id = worker_id,
                step = step,
                params = weights.len();
                "received weights"
            );

            self.grads_buf.fill(0.0);

            debug!(worker_id = worker_id, step = step; "computing gradients");
            let res =
                tokio::task::block_in_place(|| self.strategy.step(weights, &mut self.grads_buf));
            if let Err(e) = res {
                warn!(
                    worker_id = worker_id,
                    step = step;
                    "train strategy error: {}",
                    e
                );
                return Err(io::Error::new(io::ErrorKind::InvalidData, e.to_string()));
            }

            debug!(
                worker_id = worker_id,
                step = step,
                params = self.grads_buf.len();
                "sending gradient"
            );
            let out = Msg::Data(Payload::Gradient(&self.grads_buf));
            tx.send(&out).await?;
        }

        info!(worker_id = worker_id; "worker finished");
        Ok(())
    }
}
