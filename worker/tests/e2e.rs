use std::{io, num::NonZeroUsize};

use tokio::io as tokio_io;

use comms::msg::{Msg, Payload};
use ml_core::{MlError, StepStats, TrainStrategy};
use worker::{Worker, WorkerConfig};

#[derive(Debug)]
struct MockStrategy {
    n: usize,
}

impl MockStrategy {
    fn new(n: usize) -> Self {
        Self { n }
    }
}

impl TrainStrategy for MockStrategy {
    fn num_params(&self) -> usize {
        self.n
    }

    fn step(&mut self, weights: &[f32], grads: &mut [f32]) -> Result<StepStats, MlError> {
        // Worker guarantees grads are zeroed.
        // Deterministic behavior: grad[i] = 2*weights[i]
        if weights.len() != grads.len() {
            return Err(MlError::ShapeMismatch {
                what: "params",
                got: weights.len(),
                expected: grads.len(),
            });
        }
        for i in 0..weights.len() {
            grads[i] = 2.0 * weights[i];
        }
        Ok(StepStats::new(1, 0))
    }
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn worker_e2e_sends_expected_gradient() -> io::Result<()> {
    const BUF_SIZE: usize = 4096;

    let (sv_stream, wk_stream) = tokio_io::duplex(BUF_SIZE);

    // server
    let (sv_rx, sv_tx) = tokio_io::split(sv_stream);
    let (mut sv_rx, mut sv_tx) = comms::channel(sv_rx, sv_tx);

    // worker
    let (wk_rx, wk_tx) = tokio_io::split(wk_stream);
    let (wk_rx, wk_tx) = comms::channel(wk_rx, wk_tx);

    let cfg = WorkerConfig::new(0, NonZeroUsize::new(3).unwrap());
    let strat = MockStrategy::new(2);

    let worker_task = tokio::spawn(async move {
        Worker::new(cfg, strat).run(wk_rx, wk_tx).await
    });

    for step in 0..3 {
        let w = [step as f32 + 1.0, step as f32 + 2.0];
        sv_tx.send(&Msg::Data(Payload::Weights(&w))).await?;

        let msg: Msg = sv_rx.recv().await?;
        match msg {
            Msg::Data(Payload::Gradient(g)) => {
                assert_eq!(g.len(), 2);
                assert_eq!(g[0], 2.0 * w[0]);
                assert_eq!(g[1], 2.0 * w[1]);
            }
            other => panic!("unexpected msg: {other:?}"),
        }
    }

    worker_task.await.unwrap()?;
    Ok(())
}
