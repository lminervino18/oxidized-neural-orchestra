use std::io;

use comms::{
    msg::{Msg, Payload},
    OnoReceiver, OnoSender,
};
use ml_core::TrainStrategy;
use tokio::io::{AsyncRead, AsyncWrite};

use crate::config::WorkerConfig;

/// Infrastructure worker runtime.
///
/// Executes a fixed receive → compute → send loop by delegating local training
/// to an injected `TrainStrategy`.
pub struct Worker<S: TrainStrategy> {
    cfg: WorkerConfig,
    strategy: S,
    grads_buf: Vec<f32>,
}

impl<S: TrainStrategy> Worker<S> {
    /// Creates a new worker instance using the provided training strategy.
    pub fn new(cfg: WorkerConfig, strategy: S) -> Self {
        let n = strategy.num_params();
        Self {
            cfg,
            strategy,
            grads_buf: vec![0.0; n],
        }
    }

    /// Runs the worker loop for the configured number of steps.
    ///
    /// Protocol:
    /// - expects `Msg::Data(Payload::Weights)` from the receiver
    /// - emits `Msg::Data(Payload::Gradient)` to the sender
    pub async fn run<R, W>(mut self, mut rx: OnoReceiver<R>, mut tx: OnoSender<W>) -> io::Result<()>
    where
        R: AsyncRead + Unpin + Send,
        W: AsyncWrite + Unpin + Send,
    {
        for _ in 0..self.cfg.steps() {
            let msg: Msg = rx.recv().await?;

            let weights = match msg {
                Msg::Data(Payload::Weights(w)) => w,
                other => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("unexpected message: {other:?}"),
                    ))
                }
            };

            if weights.len() != self.grads_buf.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "weights length mismatch: got {}, expected {}",
                        weights.len(),
                        self.grads_buf.len()
                    ),
                ));
            }

            self.grads_buf.fill(0.0);

            let res = tokio::task::block_in_place(|| self.strategy.step(weights, &mut self.grads_buf));
            if let Err(e) = res {
                return Err(io::Error::new(io::ErrorKind::InvalidData, e.to_string()));
            }

            let out = Msg::Data(Payload::Gradient(&self.grads_buf));
            tx.send(&out).await?;
        }

        Ok(())
    }
}
