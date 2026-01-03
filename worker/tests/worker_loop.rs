use std::io;

use tokio::io as tokio_io;

use worker::{PsClient, WorkerConfig, WorkerLoop};
use comms::msg::{Msg, Payload};

#[tokio::test]
async fn worker_loop_sends_expected_gradients() -> io::Result<()> {
    const PARAMS: usize = 2;
    const STEPS: usize = 3;
    const BUF_SIZE: usize = 4096;

    // In-memory duplex link
    let (sv_stream, wk_stream) = tokio_io::duplex(BUF_SIZE);

    // Server side channel
    let (sv_rx, sv_tx) = tokio_io::split(sv_stream);
    let (mut sv_rx, mut sv_tx) = comms::channel(sv_rx, sv_tx);

    // Worker side client
    let (wk_rx, wk_tx) = tokio_io::split(wk_stream);
    let (wk_rx, wk_tx) = comms::channel(wk_rx, wk_tx);
    let client = PsClient::new(wk_rx, wk_tx);

    // Compute function: grad = weights * 10 (deterministic, easy to assert)
    let compute = |weights: &[f32], grads: &mut [f32]| {
        for (g, w) in grads.iter_mut().zip(weights) {
            *g = *w * 10.0;
        }
    };

    let cfg = WorkerConfig {
        worker_id: 0,
        num_workers: std::num::NonZeroUsize::new(1).unwrap(),
        num_params: PARAMS,
        steps: STEPS,
        microbatch_k: std::num::NonZeroUsize::new(1).unwrap(),
        prefetch: false,
    };

    // Run worker in background
    let worker_task = tokio::spawn(async move {
        let wl = WorkerLoop::new(cfg, compute);
        wl.run(client).await
    });

    // Server drives the protocol: send weights, read grads
    for step in 0..STEPS {
        let w = [step as f32 + 1.0, step as f32 + 2.0];
        sv_tx.send(&Msg::Data(Payload::Weights(&w))).await?;

        let msg: Msg = sv_rx.recv().await?;
        match msg {
            Msg::Data(Payload::Gradient(g)) => {
                assert_eq!(g.len(), PARAMS);
                assert_eq!(g[0], w[0] * 10.0);
                assert_eq!(g[1], w[1] * 10.0);
            }
            other => panic!("unexpected msg: {other:?}"),
        }
    }

    // Ensure worker finished cleanly
    let metrics = worker_task.await.unwrap()?;
    assert_eq!(metrics.steps as usize, STEPS);

    Ok(())
}
