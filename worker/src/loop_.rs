use std::io;

use tokio::task;

use super::{
    WorkerConfig, WorkerState,
    metrics::WorkerMetrics,
    net::PsClient,
};

/// Orchestrates the worker lifecycle.
///
/// This version is intentionally minimal:
/// - Uses an injected compute function to fill grads from weights.
/// - Later we will plug `data/*` and `model/*` here.
pub struct WorkerLoop<C> {
    cfg: WorkerConfig,
    state: WorkerState,
    metrics: WorkerMetrics,
    compute: C,
}

impl<C> WorkerLoop<C> {
    pub fn new(cfg: WorkerConfig, compute: C) -> Self {
        cfg.validate();
        Self {
            state: WorkerState::new(cfg.num_params),
            metrics: WorkerMetrics::default(),
            cfg,
            compute,
        }
    }

    pub fn metrics(&self) -> &WorkerMetrics {
        &self.metrics
    }
}

impl<C> WorkerLoop<C>
where
    C: Fn(&[f32], &mut [f32]) + Send + Sync + Clone + 'static,
{
    /// Runs the worker for `cfg.steps` iterations.
    pub async fn run<R, W>(mut self, mut client: PsClient<R, W>) -> io::Result<WorkerMetrics>
    where
        R: tokio::io::AsyncRead + Unpin + Send + 'static,
        W: tokio::io::AsyncWrite + Unpin + Send + 'static,
    {
        for _ in 0..self.cfg.steps {
            // 1) recv weights into persistent buffer
            let recv_res = {
                let weights = &mut self.state.weights;
                tokio::time::timeout(
                    std::time::Duration::from_secs(30),
                    async { client.recv_weights_into(weights).await },
                )
                .await
            };

            match recv_res {
                Ok(r) => r?,
                Err(_) => {
                    return Err(io::Error::new(io::ErrorKind::TimedOut, "timeout waiting for weights"));
                }
            }

            // 2) compute grads (CPU-bound)
            self.state.zero_grads();
            let weights_snapshot = self.state.weights.clone(); // temporary (we'll remove later)
            let mut grads = vec![0.0_f32; self.cfg.num_params]; // temporary (we'll remove later)
            let compute_fn = self.compute.clone();

            let compute_out = task::spawn_blocking(move || {
                (compute_fn)(&weights_snapshot, &mut grads);
                grads
            })
            .await
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("compute join error: {e}")))?;

            self.state.grads = compute_out;

            // 3) send grad
            client.send_grad(&self.state.grads).await?;

            self.state.inc_step();
            self.metrics.bump_step();
        }

        Ok(self.metrics)
    }
}


