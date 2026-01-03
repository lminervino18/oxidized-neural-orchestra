// worker/tests/worker_loop.rs

use std::{io, sync::Arc};

use tokio::io as tokio_io;

use comms::msg::{Msg, Payload};
use worker::{PsClient, WorkerConfig, WorkerLoop};
use worker::data::dataset::Batch;
use worker::model::{spec::ModelSpec, Model};

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

    // Model + fixed batch (shared safely across tasks)
    let model = Arc::new(Model::new(ModelSpec::LinearRegression1D));
    assert_eq!(model.num_params(), PARAMS);

    let batch = Arc::new(Batch::new(
        vec![1.0_f32, 2.0, 3.0, 4.0, 5.0],
        vec![3.0_f32, 5.0, 7.0, 9.0, 11.0],
    ));

    // Compute closure uses only the stable Model API.
    // Arc clones are cheap and avoid borrow-checker issues with move closures.
    let model_c = Arc::clone(&model);
    let batch_c = Arc::clone(&batch);

    let compute = move |weights: &[f32], grads: &mut [f32]| {
        model_c.grad_batch(weights, grads, &batch_c.xs, &batch_c.ys);
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

    // Server drives the protocol: send weights, read grads, validate grads.
    for step in 0..STEPS {
        let w = [step as f32 + 1.0, step as f32 + 2.0];
        sv_tx.send(&Msg::Data(Payload::Weights(&w))).await?;

        let msg: Msg = sv_rx.recv().await?;
        match msg {
            Msg::Data(Payload::Gradient(g)) => {
                assert_eq!(g.len(), PARAMS);

                let mut expected = [0.0_f32; PARAMS];
                model.grad_batch(&w, &mut expected, &batch.xs, &batch.ys);

                assert!((g[0] - expected[0]).abs() < 1e-6);
                assert!((g[1] - expected[1]).abs() < 1e-6);
            }
            other => panic!("unexpected msg: {other:?}"),
        }
    }

    // Ensure worker finished cleanly
    let metrics = worker_task.await.unwrap()?;
    assert_eq!(metrics.steps as usize, STEPS);

    Ok(())
}
