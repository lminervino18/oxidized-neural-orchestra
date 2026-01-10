use std::io;

use comms::{
    msg::{Msg, Payload},
    OnoReceiver, OnoSender,
};
use ml_core::{MlError, TrainStrategy};
use tokio::io::{AsyncRead, AsyncWrite};

use crate::config::WorkerConfig;

/// Worker runtime: receives weights, computes gradients via a strategy, and sends gradients back.
///
/// This crate is infrastructure-only: it does not implement models or datasets.
pub struct Worker<S: TrainStrategy> {
    cfg: WorkerConfig,
    strategy: S,
    grads_buf: Vec<f32>,
}

impl<S: TrainStrategy> Worker<S> {
    pub fn new(cfg: WorkerConfig, strategy: S) -> Self {
        let n = strategy.num_params();
        Self {
            cfg,
            strategy,
            grads_buf: vec![0.0; n],
        }
    }

    pub async fn run<R, W>(mut self, mut rx: OnoReceiver<R>, mut tx: OnoSender<W>) -> io::Result<()>
    where
        R: AsyncRead + Unpin + Send,
        W: AsyncWrite + Unpin + Send,
    {
        for _ in 0..self.cfg.steps() {
            // Receive message (borrowed msg/payload)
            let msg: Msg = rx.recv().await?;

            // Centralized protocol handling
            let weights = match msg {
                Msg::Data(Payload::Weights(w)) => w,
                other => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("unexpected message: {other:?}"),
                    ))
                }
            };

            // Validate boundary conditions (no asserts)
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

            // Worker guarantees grads arrive zeroed
            self.grads_buf.fill(0.0);

            // Compute without requiring 'static captures
            let res = tokio::task::block_in_place(|| {
                self.strategy.step(weights, &mut self.grads_buf)
            });

            if let Err(e) = res {
                return Err(io::Error::new(io::ErrorKind::InvalidData, e.to_string()));
            }

            // Send gradient
            let out = Msg::Data(Payload::Gradient(&self.grads_buf));
            tx.send(&out).await?;
        }

        Ok(())
    }
}
