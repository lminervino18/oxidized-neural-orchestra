use std::{io, num::NonZeroUsize};

use tokio::io as tokio_io;

use comms::msg::{Msg, Payload};
use worker::{
    PsClient, WorkerConfig, WorkerLoop,
    data::{InMemoryDataset, ShardSpec, DataLoader},
    model::{spec::ModelSpec, Model},
    train::SupervisedTrain1D,
};

#[tokio::test]
async fn worker_loop_sends_expected_gradients() -> io::Result<()> {
    const PARAMS: usize = 2;
    const STEPS: usize = 3;
    const BUF_SIZE: usize = 4096;
    const BATCH_SIZE: usize = 2;

    let microbatch_k = NonZeroUsize::new(2).unwrap();

    // In-memory duplex link
    let (sv_stream, wk_stream) = tokio_io::duplex(BUF_SIZE);

    // Server side channel
    let (sv_rx, sv_tx) = tokio_io::split(sv_stream);
    let (mut sv_rx, mut sv_tx) = comms::channel(sv_rx, sv_tx);

    // Worker side client
    let (wk_rx, wk_tx) = tokio_io::split(wk_stream);
    let (wk_rx, wk_tx) = comms::channel(wk_rx, wk_tx);
    let client = PsClient::new(wk_rx, wk_tx);

    // Dataset (global)
    let ds = InMemoryDataset::new(
        vec![1.0_f32, 2.0, 3.0, 4.0, 5.0],
        vec![3.0_f32, 5.0, 7.0, 9.0, 11.0],
    );

    let num_workers = NonZeroUsize::new(1).unwrap();
    let shard = ShardSpec::new(0, num_workers);

    // Worker loader + model + strategy
    let loader = DataLoader::new(ds.clone(), shard, BATCH_SIZE);
    let model = Model::new(ModelSpec::LinearRegression1D);
    assert_eq!(model.num_params(), PARAMS);

    let strategy = SupervisedTrain1D::new(model.clone(), loader, microbatch_k);

    let cfg = WorkerConfig {
        worker_id: 0,
        num_workers,
        num_params: PARAMS,
        steps: STEPS,
        microbatch_k,
        prefetch: false,
    };

    // Run worker in background
    let worker_task = tokio::spawn(async move {
        let wl = WorkerLoop::new(cfg, strategy);
        wl.run(client).await
    });

    // Shadow loader to generate expected microbatch consumption pattern.
    let mut shadow_loader = DataLoader::new(ds, shard, BATCH_SIZE);

    for step in 0..STEPS {
        let w = [step as f32 + 1.0, step as f32 + 2.0];
        sv_tx.send(&Msg::Data(Payload::Weights(&w))).await?;

        let msg: Msg = sv_rx.recv().await?;
        match msg {
            Msg::Data(Payload::Gradient(g)) => {
                assert_eq!(g.len(), PARAMS);

                // Compute expected from k batches with same cycling behavior.
                let mut expected = vec![0.0_f32; PARAMS];
                let mut scratch = vec![0.0_f32; PARAMS];
                let mut total_samples = 0usize;

                for _ in 0..microbatch_k.get() {
                    let batch = match shadow_loader.next_batch() {
                        Some(b) => b,
                        None => {
                            shadow_loader.reset();
                            shadow_loader.next_batch().unwrap()
                        }
                    };

                    scratch.fill(0.0);
                    model.grad_batch(&w, &mut scratch, &batch.xs, &batch.ys);

                    expected
                        .iter_mut()
                        .zip(scratch.iter())
                        .for_each(|(e, s)| *e += *s);

                    total_samples += batch.len();
                }

                let denom = total_samples as f32;
                expected.iter_mut().for_each(|e| *e /= denom);

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
