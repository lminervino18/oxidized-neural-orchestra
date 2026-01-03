// worker/src/loop_.rs

use std::io;

use tokio::task;

use crate::{
    metrics::WorkerMetrics,
    net::PsClient,
    state::WorkerState,
    train::TrainStrategy,
    WorkerConfig,
};

/// Orchestrates the worker lifecycle.
///
/// Design:
/// - Keeps persistent buffers in `WorkerState`.
/// - Receives weights into `state.weights`.
/// - Computes grads into `state.grads` with no per-step allocations.
/// - Sends `state.grads`.
///
/// Concurrency note:
/// - Compute is CPU-bound and runs on Tokio's blocking pool via `spawn_blocking`.
/// - We move the buffers out of `self` (O(1) moves) to satisfy `'static` without cloning.
/// - We also move the strategy into the blocking closure and return it back, so it can
///   keep internal state (e.g., DataLoader cursor) without locks.
pub struct WorkerLoop<S> {
    cfg: WorkerConfig,
    state: WorkerState,
    metrics: WorkerMetrics,
    strategy: S,
}

impl<S> WorkerLoop<S> {
    pub fn new(cfg: WorkerConfig, strategy: S) -> Self {
        cfg.validate();
        Self {
            state: WorkerState::new(cfg.num_params),
            metrics: WorkerMetrics::default(),
            cfg,
            strategy,
        }
    }

    pub fn metrics(&self) -> &WorkerMetrics {
        &self.metrics
    }
}

impl<S> WorkerLoop<S>
where
    S: TrainStrategy + Send + 'static,
{
    /// Runs the worker for `cfg.steps` iterations.
    pub async fn run<R, W>(mut self, mut client: PsClient<R, W>) -> io::Result<WorkerMetrics>
    where
        R: tokio::io::AsyncRead + Unpin + Send + 'static,
        W: tokio::io::AsyncWrite + Unpin + Send + 'static,
    {
        for _ in 0..self.cfg.steps {
            // 1) Receive weights into persistent buffer.
            client.recv_weights_into(&mut self.state.weights).await?;

            // 2) Compute gradients (CPU-bound) without per-step allocations/clones.
            self.state.zero_grads();

            // Move buffers + strategy out to satisfy `'static` for spawn_blocking, without copying.
            let weights = std::mem::take(&mut self.state.weights);
            let mut grads = std::mem::take(&mut self.state.grads);
            let mut strategy = self.strategy;

            // Maintain invariants: lengths must match the declared parameter count.
            debug_assert_eq!(weights.len(), self.cfg.num_params);
            debug_assert_eq!(grads.len(), self.cfg.num_params);

            // Run compute on the blocking pool.
            let (strategy_back, weights_back, grads_back) = task::spawn_blocking(move || {
                strategy.compute_step(&weights, &mut grads);
                (strategy, weights, grads)
            })
            .await
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("compute join error: {e}")))?;

            // Restore state (no allocations).
            self.strategy = strategy_back;
            self.state.weights = weights_back;
            self.state.grads = grads_back;

            // 3) Send gradients.
            client.send_grad(&self.state.grads).await?;

            self.state.inc_step();
            self.metrics.bump_step();
        }

        Ok(self.metrics)
    }
}
