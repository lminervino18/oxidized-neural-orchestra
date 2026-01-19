use std::{io, num::NonZeroUsize, time::Duration};

use tokio::io as tokio_io;
use tokio::time::timeout;

use comms::msg::{Command, Msg, Payload};
use comms::specs::worker::{ExecutionSpec, WorkerSpec};
use comms::specs::strategy::{StrategySpec};
use ml_core::{MlError, StepStats, TrainStrategy};

struct NoopStrategy;

impl TrainStrategy for NoopStrategy {
    fn step(&mut self, _weights: &[f32], _grads: &mut [f32]) -> Result<StepStats, MlError> {
        Ok(StepStats::new(1, 0))
    }
}

fn mk_spec(steps: usize, num_params: usize) -> WorkerSpec {
    WorkerSpec {
        worker_id: 0,
        steps: NonZeroUsize::new(steps).unwrap(),
        num_params: NonZeroUsize::new(num_params).unwrap(),
        strategy: StrategySpec {
            kind: "noop".to_string(),
            params: serde_json::Value::Null,
        },
        artifacts: Default::default(),
        execution: ExecutionSpec::Default,
        seed: None,
    }
}

async fn assert_no_gradient_received<R: tokio::io::AsyncRead + Unpin>(
    sv_rx: &mut comms::OnoReceiver<R>,
) {
    let recv_res = timeout(Duration::from_millis(50), sv_rx.recv::<Msg>()).await;

    match recv_res {
        Err(_) => {}
        Ok(Err(_)) => {}
        Ok(Ok(Msg::Data(Payload::Gradient(_)))) => {
            panic!("server unexpectedly received a Gradient message");
        }
        Ok(Ok(_)) => {}
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn worker_rejects_wrong_weight_length() -> io::Result<()> {
    let (sv_stream, wk_stream) = tokio_io::duplex(4096);

    let (sv_rx, sv_tx) = tokio_io::split(sv_stream);
    let (mut sv_rx, mut sv_tx) = comms::channel(sv_rx, sv_tx);

    let (wk_rx, wk_tx) = tokio_io::split(wk_stream);
    let (wk_rx, wk_tx) = comms::channel(wk_rx, wk_tx);

    let spec = mk_spec(1, 2);
    sv_tx.send(&Msg::Control(Command::CreateWorker(spec))).await?;

    let worker_task = tokio::spawn(async move {
        let factory = |spec: &WorkerSpec| -> io::Result<NoopStrategy> {
            match spec.strategy.kind.as_str() {
                "noop" => Ok(NoopStrategy),
                other => Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unknown strategy kind: {other}"),
                )),
            }
        };

        worker::run_bootstrapped(wk_rx, wk_tx, factory).await
    });

    let mut w = [1.0_f32, 2.0, 3.0];
    sv_tx.send(&Msg::Data(Payload::Weights(&mut w))).await?;

    let res = worker_task.await.unwrap();
    assert!(res.is_err());

    assert_no_gradient_received(&mut sv_rx).await;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn worker_rejects_unexpected_message() -> io::Result<()> {
    let (sv_stream, wk_stream) = tokio_io::duplex(4096);

    let (sv_rx, sv_tx) = tokio_io::split(sv_stream);
    let (mut sv_rx, mut sv_tx) = comms::channel(sv_rx, sv_tx);

    let (wk_rx, wk_tx) = tokio_io::split(wk_stream);
    let (wk_rx, wk_tx) = comms::channel(wk_rx, wk_tx);

    let spec = mk_spec(1, 2);
    sv_tx.send(&Msg::Control(Command::CreateWorker(spec))).await?;

    let worker_task = tokio::spawn(async move {
        let factory = |spec: &WorkerSpec| -> io::Result<NoopStrategy> {
            match spec.strategy.kind.as_str() {
                "noop" => Ok(NoopStrategy),
                other => Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unknown strategy kind: {other}"),
                )),
            }
        };

        worker::run_bootstrapped(wk_rx, wk_tx, factory).await
    });

    let g = [0.1_f32, 0.2];
    sv_tx.send(&Msg::Data(Payload::Gradient(&g))).await?;

    let res = worker_task.await.unwrap();
    assert!(res.is_err());

    assert_no_gradient_received(&mut sv_rx).await;
    Ok(())
}
