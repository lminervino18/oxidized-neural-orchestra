use std::io;
use std::time::Instant;

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
/// - We move buffers out of `self` (O(1) moves) to satisfy `'static` without cloning.
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
            // 1) RECV
            let t0 = Instant::now();
            client.recv_weights_into(&mut self.state.weights).await?;
            self.metrics.recv_time += t0.elapsed();

            // 2) COMPUTE (blocking pool)
            self.state.zero_grads();

            // Move buffers and strategy out (O(1)), to satisfy `'static` for spawn_blocking.
            let weights = std::mem::take(&mut self.state.weights);
            let mut grads = std::mem::take(&mut self.state.grads);
            let mut strategy = self.strategy;

            debug_assert_eq!(weights.len(), self.cfg.num_params);
            debug_assert_eq!(grads.len(), self.cfg.num_params);

            let t1 = Instant::now();
            let (strategy_back, weights_back, grads_back) = task::spawn_blocking(move || {
                strategy.compute_step(&weights, &mut grads);
                (strategy, weights, grads)
            })
            .await
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("compute join error: {e}")))?;
            self.metrics.compute_time += t1.elapsed();

            self.strategy = strategy_back;
            self.state.weights = weights_back;
            self.state.grads = grads_back;

            // Update counters from the strategy-reported stats
            let stats = self.strategy.last_step_stats();
            self.metrics.add_microbatches(stats.microbatches);
            self.metrics.add_samples(stats.samples);

            // 3) SEND
            let t2 = Instant::now();
            client.send_grad(&self.state.grads).await?;
            self.metrics.send_time += t2.elapsed();

            self.state.inc_step();
            self.metrics.bump_step();
        }

        Ok(self.metrics)
    }
}
