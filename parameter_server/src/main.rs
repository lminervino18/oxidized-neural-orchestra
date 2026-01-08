mod parameters;
mod server;
mod training;

use std::{io, num::NonZeroUsize};

use tokio::net::TcpListener;

use crate::{
    parameters::{ParameterStore, optimization::GradientDescent, weight_gen::ConstWeightGen},
    server::ParameterServer,
    training::BarrierSyncTrainer,
};

#[tokio::main]
async fn main() -> io::Result<()> {
    const ADDR: &'static str = "127.0.0.1:8765";
    const WORKERS: usize = 1;
    const PARAMS: usize = 2;
    const SHARD_AMOUNT: NonZeroUsize = NonZeroUsize::new(2).unwrap();
    const EPOCHS: usize = 500;
    const LR: f32 = 0.01;

    let weight_gen = ConstWeightGen::new(0., PARAMS);
    let optimizer_factory = |_| GradientDescent::new(LR);
    let store = ParameterStore::new(PARAMS, SHARD_AMOUNT, weight_gen, optimizer_factory);
    let trainer = BarrierSyncTrainer::new(WORKERS, store);
    let mut pserver = ParameterServer::new(PARAMS, EPOCHS, trainer);

    let list = TcpListener::bind(ADDR).await?;

    for _ in 0..WORKERS {
        let (stream, _) = list.accept().await?;
        let (rx, tx) = stream.into_split();
        let (rx, tx) = comms::channel(rx, tx);
        pserver.spawn(rx, tx);
    }

    pserver.run().await.into_iter().collect::<io::Result<_>>()
}
