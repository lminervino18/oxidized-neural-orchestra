use std::{io, num::NonZeroUsize};

use tokio::io as tokio_io;

use comms::msg::{Command, Msg, Payload};
use comms::specs::worker::{ExecutionSpec, WorkerSpec};
use comms::specs::strategy::{StrategySpec};
use ml_core::{MlError, StepStats, TrainStrategy};

fn mk_spec(steps: usize, num_params: usize) -> WorkerSpec {
    WorkerSpec {
        worker_id: 0,
        steps: NonZeroUsize::new(steps).unwrap(),
        num_params: NonZeroUsize::new(num_params).unwrap(),
        strategy: StrategySpec {
            kind: "mock".to_string(),
            params: serde_json::Value::Null,
        },
        artifacts: Default::default(),
        execution: ExecutionSpec::Default,
        seed: None,
    }
}

#[derive(Debug)]
struct MockStrategy;

impl TrainStrategy for MockStrategy {
    fn step(&mut self, weights: &[f32], grads: &mut [f32]) -> Result<StepStats, MlError> {
        if weights.len() != grads.len() {
            return Err(MlError::ShapeMismatch {
                what: "params",
                got: weights.len(),
                expected: grads.len(),
            });
        }

        for (g, w) in grads.iter_mut().zip(weights.iter()) {
            *g = 2.0 * *w;
        }

        Ok(StepStats::new(1, 0))
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn worker_e2e_sends_expected_gradient() -> io::Result<()> {
    const BUF_SIZE: usize = 4096;
    const STEPS: usize = 3;
    const PARAMS: usize = 2;

    let (sv_stream, wk_stream) = tokio_io::duplex(BUF_SIZE);

    let (sv_rx, sv_tx) = tokio_io::split(sv_stream);
    let (mut sv_rx, mut sv_tx) = comms::channel(sv_rx, sv_tx);

    let (wk_rx, wk_tx) = tokio_io::split(wk_stream);
    let (wk_rx, wk_tx) = comms::channel(wk_rx, wk_tx);

    let spec = mk_spec(STEPS, PARAMS);
    sv_tx.send(&Msg::Control(Command::CreateWorker(spec))).await?;

    let worker_task = tokio::spawn(async move {
        let factory = |spec: &WorkerSpec| -> io::Result<MockStrategy> {
            match spec.strategy.kind.as_str() {
                "mock" => Ok(MockStrategy),
                other => Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unknown strategy kind: {other}"),
                )),
            }
        };

        worker::run_bootstrapped(wk_rx, wk_tx, factory).await
    });

    for step in 0..STEPS {
        let mut w = [step as f32 + 1.0, step as f32 + 2.0];
        sv_tx.send(&Msg::Data(Payload::Weights(&mut w))).await?;

        let msg: Msg = sv_rx.recv().await?;
        match msg {
            Msg::Data(Payload::Gradient(g)) => {
                assert_eq!(g.len(), PARAMS);
                assert_eq!(g[0], 2.0 * w[0]);
                assert_eq!(g[1], 2.0 * w[1]);
            }
            other => panic!("unexpected msg: {other:?}"),
        }
    }

    worker_task.await.unwrap()?;
    Ok(())
}
