use std::num::NonZeroUsize;

use comms::{
    msg::{Msg, Payload},
    OnoReceiver, OnoSender,
};
use log::{debug, error, info, warn};
use machine_learning::TrainStrategy;
use tokio::io::{AsyncRead, AsyncWrite};

use crate::{config::WorkerConfig, error::WorkerError};

/// Infrastructure worker runtime.
pub struct Worker<S: TrainStrategy> {
    worker_id: usize,
    cfg: WorkerConfig,
    strategy: S,
    num_params: NonZeroUsize,
    grads_buf: Vec<f32>,
}

impl<S: TrainStrategy> Worker<S> {
    /// Creates a new `Worker`.
    ///
    /// # Args
    /// * `worker_id` - Worker identifier used for observability.
    /// * `cfg` - Immutable execution bounds.
    /// * `num_params` - Expected parameter count for `weights` and `grads`.
    /// * `strategy` - Training strategy used to compute gradients.
    ///
    /// # Returns
    /// A new `Worker` instance.
    pub fn new(worker_id: usize, cfg: WorkerConfig, num_params: NonZeroUsize, strategy: S) -> Self {
        let n = num_params.get();
        Self {
            worker_id,
            cfg,
            strategy,
            num_params,
            grads_buf: vec![0.0; n],
        }
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
    /// Returns `WorkerError` on I/O failures, protocol violations, or ML strategy failures.
    pub async fn run<R, W>(
        mut self,
        mut rx: OnoReceiver<R>,
        mut tx: OnoSender<W>,
    ) -> Result<(), WorkerError>
    where
        R: AsyncRead + Unpin + Send,
        W: AsyncWrite + Unpin + Send,
    {
        let worker_id = self.worker_id;
        let expected = self.num_params.get();

        info!(worker_id = worker_id; "worker starting");

        for step in 0..self.cfg.steps() {
            debug!(worker_id = worker_id, step = step; "waiting for weights");
            let msg: Msg = rx.recv().await?;

            let weights = match msg {
                Msg::Data(Payload::Params(w)) => w,
                other => {
                    let got = msg_kind(&other);
                    error!(worker_id = worker_id, step = step, got = got; "unexpected message");
                    return Err(WorkerError::UnexpectedMessage { step, got });
                }
            };

            if weights.len() != expected {
                let got = weights.len();

                error!(
                    worker_id = worker_id,
                    step = step,
                    got = got,
                    expected = expected;
                    "weights length mismatch"
                );

                return Err(WorkerError::WeightsLengthMismatch {
                    step,
                    got,
                    expected,
                });
            }

            self.grads_buf.fill(0.0);

            debug!(worker_id = worker_id, step = step; "computing gradients");
            let res =
                tokio::task::block_in_place(|| self.strategy.step(weights, &mut self.grads_buf));
            if let Err(e) = res {
                warn!(worker_id = worker_id, step = step; "train strategy error: {e}");
                return Err(WorkerError::TrainFailed { step, source: e });
            }

            debug!(worker_id = worker_id, step = step; "sending gradient");
            let out = Msg::Data(Payload::Grad(&self.grads_buf));
            tx.send(&out).await?;
        }

        info!(worker_id = worker_id; "worker finished");
        Ok(())
    }
}

fn msg_kind(msg: &Msg<'_>) -> &'static str {
    match msg {
        Msg::Control(_) => "control",
        Msg::Err(_) => "err",
        Msg::Data(Payload::Grad(_)) => "data/gradient",
        Msg::Data(Payload::Params(_)) => "data/weights",
    }
}
