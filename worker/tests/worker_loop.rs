use std::{io, sync::Arc};

use tokio::io as tokio_io;

use comms::msg::{Msg, Payload};
use worker::{PsClient, WorkerConfig, WorkerLoop};
use worker::model::{layout::ParameterLayout, ops::linreg_mse_grad_batch, spec::ModelSpec};

#[tokio::test]
async fn worker_loop_sends_expected_gradients() -> io::Result<()> {
    const PARAMS: usize = 2;
    const STEPS: usize = 3;
    const BUF_SIZE: usize = 4096;

    let (sv_stream, wk_stream) = tokio_io::duplex(BUF_SIZE);

    let (sv_rx, sv_tx) = tokio_io::split(sv_stream);
    let (mut sv_rx, mut sv_tx) = comms::channel(sv_rx, sv_tx);

    let (wk_rx, wk_tx) = tokio_io::split(wk_stream);
    let (wk_rx, wk_tx) = comms::channel(wk_rx, wk_tx);
    let client = PsClient::new(wk_rx, wk_tx);

    // Shared model + fixed batch
    let spec = ModelSpec::LinearRegression1D;
    let layout = Arc::new(ParameterLayout::new(spec));
    layout.validate(PARAMS);

    let xs = Arc::new(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0]);
    let ys = Arc::new(vec![3.0_f32, 5.0, 7.0, 9.0, 11.0]);

    // Worker compute closure (moves Arcs cheaply)
    let layout_c = Arc::clone(&layout);
    let xs_c = Arc::clone(&xs);
    let ys_c = Arc::clone(&ys);

    let compute = move |weights: &[f32], grads: &mut [f32]| {
        linreg_mse_grad_batch(&layout_c, weights, grads, &xs_c, &ys_c);
    };

    let cfg = WorkerConfig {
        worker_id: 0,
        num_workers: std::num::NonZeroUsize::new(1).unwrap(),
        num_params: PARAMS,
        steps: STEPS,
        microbatch_k: std::num::NonZeroUsize::new(1).unwrap(),
        prefetch: false,
    };

    let worker_task = tokio::spawn(async move {
        let wl = WorkerLoop::new(cfg, compute);
        wl.run(client).await
    });

    // Server drives protocol and validates grads.
    for step in 0..STEPS {
        let w = [step as f32 + 1.0, step as f32 + 2.0];
        sv_tx.send(&Msg::Data(Payload::Weights(&w))).await?;

        let msg: Msg = sv_rx.recv().await?;
        match msg {
            Msg::Data(Payload::Gradient(g)) => {
                assert_eq!(g.len(), PARAMS);

                let mut expected = [0.0_f32; PARAMS];
                linreg_mse_grad_batch(&layout, &w, &mut expected, &xs, &ys);

                assert!((g[0] - expected[0]).abs() < 1e-6);
                assert!((g[1] - expected[1]).abs() < 1e-6);
            }
            other => panic!("unexpected msg: {other:?}"),
        }
    }

    let metrics = worker_task.await.unwrap()?;
    assert_eq!(metrics.steps as usize, STEPS);

    Ok(())
}
