mod parameters;
mod server;
mod training;

use std::num::NonZeroUsize;

use comms::msg::{Msg, Payload};
use tokio::{io, net::TcpListener, task::JoinSet};

use crate::{
    parameters::{ParameterStore, optimization::GradientDescent},
    server::ParameterServer,
    training::BarrierSync,
};

// use std::io;

// use std::num::NonZeroUsize;

// use comms::msg::{Msg, Payload};
// use tokio::net::TcpListener;

// use crate::{
//     parameters::{ParameterStore, optimization::GradientDescent},
//     server::ParameterServer,
//     training::BarrierSync,
// };

// #[tokio::main]
// async fn main() -> io::Result<()> {
//     const ADDR: &'static str = "127.0.0.1:8765";
//     const WORKERS: usize = 1;
//     const PARAMS: usize = 2;
//     const SHARD_AMOUNT: NonZeroUsize = NonZeroUsize::new(2).unwrap();
//     const EPOCHS: usize = 500;
//     const LR: f32 = 0.01;

//     let store = ParameterStore::new(PARAMS, SHARD_AMOUNT, |_| GradientDescent::new(LR));
//     let trainer = BarrierSync::new(store, WORKERS);
//     let mut pserver = ParameterServer::new(PARAMS, EPOCHS, trainer);
//     let list = TcpListener::bind(ADDR).await?;

//     for _ in 0..WORKERS {
//         let (stream, _) = list.accept().await?;
//         let (rx, tx) = stream.into_split();
//         let (rx, tx) = comms::channel(rx, tx);
//         pserver.spawn(rx, tx);
//     }

//     pserver.run().await.into_iter().collect::<io::Result<_>>()
// }

#[tokio::main]
async fn main() -> io::Result<()> {
    // 1. Constants
    const WORKERS: usize = 4;
    const PARAMS: usize = 2;
    const SHARD_AMOUNT: NonZeroUsize = NonZeroUsize::new(2).unwrap();
    const EPOCHS: usize = 1000;
    const LR: f32 = 0.01;
    const BUF_SIZE: usize = PARAMS * 2 * size_of::<f32>();

    // 2. Connection mocks
    let mut sv_streams = Vec::with_capacity(WORKERS);
    let mut wk_streams = Vec::with_capacity(WORKERS);

    for _ in 0..WORKERS {
        let (sv_stream, wk_stream) = io::duplex(BUF_SIZE);

        let (rx, tx) = io::split(sv_stream);
        let ono_chan = comms::channel(rx, tx);
        sv_streams.push(ono_chan);

        let (rx, tx) = io::split(wk_stream);
        let ono_chan = comms::channel(rx, tx);
        wk_streams.push(ono_chan);
    }

    // 3. Setup Parameter Server
    let store = ParameterStore::new(PARAMS, SHARD_AMOUNT, |_| GradientDescent::new(LR));
    let trainer = BarrierSync::new(store, WORKERS);
    let mut pserver = ParameterServer::new(PARAMS, EPOCHS, trainer);

    for (rx, tx) in sv_streams {
        pserver.spawn(rx, tx);
    }

    // 4. Setup mocked Workers
    let mut worker_tasks = JoinSet::new();
    let x = [1., 2., 3., 4., 5.];
    let y = [3., 5., 7., 9., 11.];

    for (mut rx, mut tx) in wk_streams {
        let handle = async move {
            let mut grad = vec![0.; PARAMS];
            let n = x.len();
            let two_over_n = 2. / n as f32;

            for _ in 0..EPOCHS {
                let msg: Msg = rx.recv().await?;

                if let Msg::Data(Payload::Weights(weights)) = msg {
                    for i in 0..n as usize {
                        let pred = weights[0] * x[i] + weights[1];
                        let err = pred - y[i];

                        grad[0] += two_over_n * err * x[i];
                        grad[1] += two_over_n * err;
                    }

                    let msg = Msg::Data(Payload::Gradient(&grad));
                    tx.send(&msg).await?;
                    grad.fill(0.);

                    println!("weights: {weights:?}");
                }
            }

            Ok::<_, io::Error>(())
        };

        worker_tasks.spawn(handle);
    }

    // 5. Run the server and the workers.
    tokio::join!(pserver.run(), worker_tasks.join_all());
    Ok(())
}
