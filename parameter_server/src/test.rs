#![cfg(test)]

use std::num::NonZeroUsize;

use comms::{
    OnoReceiver, OnoSender,
    msg::{Command, Msg, Payload},
};
use tokio::io::{self, AsyncRead, AsyncWrite, DuplexStream, ReadHalf, WriteHalf};

use crate::{
    initialization::ConstParamGen,
    optimization::GradientDescent,
    service::ParameterServer,
    storage::{BlockingStore, StoreHandle},
    synchronization::BarrierSync,
};

fn channel_pair() -> (
    (
        OnoReceiver<ReadHalf<DuplexStream>>,
        OnoSender<WriteHalf<DuplexStream>>,
    ),
    (
        OnoReceiver<ReadHalf<DuplexStream>>,
        OnoSender<WriteHalf<DuplexStream>>,
    ),
) {
    let (stream1, stream2) = io::duplex(4096);
    let (rx1, tx1) = io::split(stream1);
    let (rx2, tx2) = io::split(stream2);
    let chan1 = comms::channel(rx1, tx1);
    let chan2 = comms::channel(rx2, tx2);
    (chan1, chan2)
}

async fn mock_lineal_worker<R, W>(
    mut rx: OnoReceiver<R>,
    mut tx: OnoSender<W>,
    max_epochs: usize,
    nparams: usize,
) -> io::Result<()>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let mut rx_buf = vec![128];
    let mut grad = vec![0.0; nparams];

    for _ in 0..max_epochs {
        match rx.recv_into(&mut rx_buf).await? {
            Msg::Data(Payload::Params(params)) => {
                for (g, p) in grad.iter_mut().zip(params) {
                    *g = *p - 1.0;
                }

                let msg = Msg::Data(Payload::Grad(&grad));
                tx.send(&msg).await?
            }
            _ => {}
        }
    }

    let msg = Msg::Control(Command::Disconnect);
    tx.send(&msg).await?;

    while !matches!(
        rx.recv_into(&mut rx_buf).await?,
        Msg::Control(Command::Disconnect)
    ) {}

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_lineal_convergence() -> io::Result<()> {
    let ((wk_rx, wk_tx), (sv_rx, sv_tx)) = channel_pair();

    const MAX_EPOCHS: usize = 100;
    const NPARAMS: usize = 2;

    let shard_size = NonZeroUsize::new(1).unwrap();
    let param_gen = ConstParamGen::new(0.5, NPARAMS);
    let optimizer_factory = |_| GradientDescent::new(0.1);
    let store = BlockingStore::new(shard_size, param_gen, optimizer_factory);
    let handle = StoreHandle::new(store);
    let synchronizer = BarrierSync::new(1);
    let mut server = ParameterServer::new(handle, synchronizer);
    server.spawn(sv_rx, sv_tx);

    let worker_fut = mock_lineal_worker(wk_rx, wk_tx, MAX_EPOCHS, NPARAMS);
    let (_, params) = tokio::try_join!(worker_fut, server.run())?;
    println!("params: {params:?}");
    Ok(())
}
