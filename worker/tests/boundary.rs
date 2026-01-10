use std::{io, num::NonZeroUsize, time::Duration};

use tokio::io as tokio_io;
use tokio::time::timeout;

use comms::msg::{Msg, Payload};
use ml_core::{MlError, StepStats, TrainStrategy};
use worker::{Worker, WorkerConfig};

struct NoopStrategy;

impl TrainStrategy for NoopStrategy {
    fn step(&mut self, _weights: &[f32], _grads: &mut [f32]) -> Result<StepStats, MlError> {
        Ok(StepStats::new(1, 0))
    }
}

async fn assert_no_gradient_received<R: tokio::io::AsyncRead + Unpin>(
    sv_rx: &mut comms::OnoReceiver<R>,
) {
    let recv_res = timeout(Duration::from_millis(50), sv_rx.recv::<Msg>()).await;

    match recv_res {
        Err(_) => {
            // Timeout: no message observed, OK.
        }
        Ok(Err(_)) => {
            // Channel closed or invalid frame, OK for this test.
        }
        Ok(Ok(Msg::Data(Payload::Gradient(_)))) => {
            panic!("server unexpectedly received a Gradient message");
        }
        Ok(Ok(_)) => {
            // Some other message kind (future-proof), OK.
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn worker_rejects_wrong_weight_length() -> io::Result<()> {
    let (sv_stream, wk_stream) = tokio_io::duplex(4096);

    let (sv_rx, sv_tx) = tokio_io::split(sv_stream);
    let (mut sv_rx, mut sv_tx) = comms::channel(sv_rx, sv_tx);

    let (wk_rx, wk_tx) = tokio_io::split(wk_stream);
    let (wk_rx, wk_tx) = comms::channel(wk_rx, wk_tx);

    let cfg = WorkerConfig::new(0, NonZeroUsize::new(1).unwrap());
    let num_params = NonZeroUsize::new(2).unwrap();
    let strat = NoopStrategy;

    let worker_task = tokio::spawn(async move {
        Worker::new(cfg, num_params, strat).run(wk_rx, wk_tx).await
    });

    let w = [1.0_f32, 2.0, 3.0];
    sv_tx.send(&Msg::Data(Payload::Weights(&w))).await?;

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

    let cfg = WorkerConfig::new(0, NonZeroUsize::new(1).unwrap());
    let num_params = NonZeroUsize::new(2).unwrap();
    let strat = NoopStrategy;

    let worker_task = tokio::spawn(async move {
        Worker::new(cfg, num_params, strat).run(wk_rx, wk_tx).await
    });

    let g = [0.1_f32, 0.2];
    sv_tx.send(&Msg::Data(Payload::Gradient(&g))).await?;

    let res = worker_task.await.unwrap();
    assert!(res.is_err());

    assert_no_gradient_received(&mut sv_rx).await;
    Ok(())
}
