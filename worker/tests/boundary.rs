use std::{io, num::NonZeroUsize};

use tokio::io as tokio_io;

use comms::msg::{Msg, Payload};
use ml_core::{MlError, StepStats, TrainStrategy};
use worker::{Worker, WorkerConfig};

struct NoopStrategy { n: usize }
impl TrainStrategy for NoopStrategy {
    fn num_params(&self) -> usize { self.n }
    fn step(&mut self, _weights: &[f32], _grads: &mut [f32]) -> Result<StepStats, MlError> {
        Ok(StepStats::new(1, 0))
    }
}

#[tokio::test]
async fn worker_rejects_wrong_weight_length() -> io::Result<()> {
    let (sv_stream, wk_stream) = tokio_io::duplex(4096);

    let (sv_rx, sv_tx) = tokio_io::split(sv_stream);
    let (mut sv_rx, mut sv_tx) = comms::channel(sv_rx, sv_tx);

    let (wk_rx, wk_tx) = tokio_io::split(wk_stream);
    let (wk_rx, wk_tx) = comms::channel(wk_rx, wk_tx);

    let cfg = WorkerConfig::new(0, NonZeroUsize::new(1).unwrap());
    let strat = NoopStrategy { n: 2 };

    let worker_task = tokio::spawn(async move {
        Worker::new(cfg, strat).run(wk_rx, wk_tx).await
    });

    // send wrong length weights
    let w = [1.0_f32, 2.0, 3.0];
    sv_tx.send(&Msg::Data(Payload::Weights(&w))).await?;

    // worker should error and exit
    let res = worker_task.await.unwrap();
    assert!(res.is_err());

    // server should not receive a gradient
    let _ = sv_rx;

    Ok(())
}

#[tokio::test]
async fn worker_rejects_unexpected_message() -> io::Result<()> {
    let (sv_stream, wk_stream) = tokio_io::duplex(4096);

    let (sv_rx, sv_tx) = tokio_io::split(sv_stream);
    let (mut sv_rx, mut sv_tx) = comms::channel(sv_rx, sv_tx);

    let (wk_rx, wk_tx) = tokio_io::split(wk_stream);
    let (wk_rx, wk_tx) = comms::channel(wk_rx, wk_tx);

    let cfg = WorkerConfig::new(0, NonZeroUsize::new(1).unwrap());
    let strat = NoopStrategy { n: 2 };

    let worker_task = tokio::spawn(async move {
        Worker::new(cfg, strat).run(wk_rx, wk_tx).await
    });

    // send Gradient as if it were a server message (invalid for worker receive path)
    let g = [0.1_f32, 0.2];
    sv_tx.send(&Msg::Data(Payload::Gradient(&g))).await?;

    let res = worker_task.await.unwrap();
    assert!(res.is_err());

    // server should not receive anything
    let _ = sv_rx;

    Ok(())
}
