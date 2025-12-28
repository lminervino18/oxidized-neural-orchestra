mod execution;
mod parameters;

use std::{borrow::Cow, num::NonZeroUsize, time::Instant};

use comms::msg::{Msg, Payload};
use tokio::{
    io,
    task::{self, JoinSet},
};

use crate::{
    execution::BulkSync,
    parameters::{Optimizer, ParameterStore},
};

#[derive(Clone)]
struct TestOptimizer {}

impl Optimizer for TestOptimizer {
    fn update_weights(&mut self, weights: &mut [f32], grad: &[f32]) {
        for (w, g) in weights.iter_mut().zip(grad) {
            *w += *g;
        }
    }
}

#[tokio::main]
async fn main() {
    // 1.1 Setup constants
    const PARAMETERS: usize = 23_500_000;
    const SHARDS: NonZeroUsize = NonZeroUsize::new(PARAMETERS / 10_000).unwrap();
    const WORKERS: usize = 4;
    const BUF_SIZE: usize = 5738 * 4096;
    const EPOCHS: usize = 2;

    // 1.2 Setup mocked communication buffers
    let (mut sv_streams, mut wk_streams) = (Vec::new(), Vec::new());

    for _ in 0..WORKERS {
        let (sv_stream, wk_stream) = io::duplex(BUF_SIZE);

        let (rx, tx) = io::split(sv_stream);
        let ono_channel = comms::channel(rx, tx);
        sv_streams.push(ono_channel);

        let (rx, tx) = io::split(wk_stream);
        let ono_channel = comms::channel(rx, tx);
        wk_streams.push(ono_channel);
    }

    // 2. Run Parameter Server
    let store = ParameterStore::new(PARAMETERS, SHARDS, TestOptimizer {});
    let mut executor = BulkSync::new(store, WORKERS);

    for (mut rx, mut tx) in sv_streams {
        executor.spawn(async move |handle, barrier| {
            let mut buf = vec![0.; PARAMETERS];

            for _ in 0..EPOCHS {
                // Wait for gradient
                match rx.recv::<Msg>().await.unwrap() {
                    Msg::Data(Payload::Gradient(grad)) => handle.accumulate(&grad),
                    _ => unreachable!(),
                }

                // Wait till every worker has accumulated it's gradient
                // Select one to update the weights of the server
                if barrier.wait().await.is_leader() {
                    task::block_in_place(|| handle.update_weights());
                }

                // Wait until the leader updates the weights
                barrier.wait().await;

                // Get the updated weights
                task::block_in_place(|| handle.pull_weights(&mut buf));

                // Send the new weights to the worker nodes
                let msg = Msg::Data(Payload::Weights(Cow::Borrowed(&buf)));
                tx.send(&msg).await.unwrap();
            }
        });
    }

    // 3. Run mocked workers
    let mut workers_tasks = JoinSet::new();

    // Send the same gradient over and over and print the new weights each time
    for (id, (mut rx, mut tx)) in wk_streams.into_iter().enumerate() {
        workers_tasks.spawn(async move {
            let grad: Vec<_> = (0..PARAMETERS).map(|i| i as f32).collect();

            for epoch in 0..EPOCHS {
                let msg = Msg::Data(Payload::Gradient(Cow::Borrowed(&grad)));
                tx.send(&msg).await.unwrap();

                match rx.recv::<Msg>().await.unwrap() {
                    Msg::Data(Payload::Weights(..)) => {
                        println!("{id}: Got the new weights for epoch {epoch}")
                    }
                    _ => unreachable!(),
                }
            }
        });
    }

    // 4. Execute the whole thing
    let start = Instant::now();
    tokio::join!(executor.join_all(), workers_tasks.join_all());
    let elapsed = Instant::now().duration_since(start);

    println!("took {} seconds for RTT", elapsed.as_secs_f32());
}
