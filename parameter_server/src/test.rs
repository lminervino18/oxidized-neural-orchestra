#![cfg(test)]

use std::{env, num::NonZeroUsize};

use comms::{OrchHandle, ParamServerHandle, WorkerEvent, WorkerHandle, floats::FloatPositive};
use machine_learning::{initialization::ConstParamGen, optimization::GradientDescent};
use tokio::io::{self, AsyncRead, AsyncWrite, DuplexStream, ReadHalf, WriteHalf};

use crate::{
    service::ParameterServer,
    storage::{BlockingStore, StoreHandle},
    synchronization::BarrierSync,
};

fn channel_pair() -> (
    (ReadHalf<DuplexStream>, WriteHalf<DuplexStream>),
    (ReadHalf<DuplexStream>, WriteHalf<DuplexStream>),
) {
    let (stream1, stream2) = io::duplex(4096);
    let chan1 = io::split(stream1);
    let chan2 = io::split(stream2);
    (chan1, chan2)
}

async fn mock_lineal_worker<R, W>(
    rx: R,
    tx: W,
    orch_rx: R,
    orch_tx: W,
    max_epochs: usize,
    nparams: usize,
) -> io::Result<()>
where
    R: AsyncRead + Unpin + Send,
    W: AsyncWrite + Unpin + Send,
{
    let mut grad = vec![0.0; nparams];
    let transport = comms::build_simple_transport(rx, tx);
    let mut server_handle = ParamServerHandle::new(0, transport);

    let transport = comms::build_simple_transport(orch_rx, orch_tx);
    let mut orch_handle = OrchHandle::new(transport);

    for _ in 0..max_epochs {
        let params = server_handle.pull_params().await?;
        for (g, p) in grad.iter_mut().zip(params) {
            *g = *p - 1.0;
        }

        server_handle.push_grad(&grad).await?;
    }

    server_handle.disconnect().await?;
    orch_handle.done().await?;
    orch_handle.disconnect().await?;
    Ok(())
}

async fn mock_orch<R, W>(wk_rx: R, wk_tx: W, sv_rx: R, sv_tx: W) -> io::Result<Vec<f32>>
where
    R: AsyncRead + Unpin + Send,
    W: AsyncWrite + Unpin + Send,
{
    let transport = comms::build_simple_transport(wk_rx, wk_tx);
    let mut worker_handle = WorkerHandle::new(0, transport);

    let transport = comms::build_simple_transport(sv_rx, sv_tx);
    let mut server_handle = ParamServerHandle::new(0, transport);

    while !matches!(worker_handle.recv_event().await?, WorkerEvent::Done) {}
    while !matches!(worker_handle.recv_event().await?, WorkerEvent::Disconnect) {}

    let params = server_handle.pull_params().await?.to_vec();
    server_handle.disconnect().await?;
    Ok(params)
}

#[tokio::test(flavor = "multi_thread")]
async fn test_lineal_convergence() -> io::Result<()> {
    unsafe { env::set_var("RUST_BACKTRACE", "1") };

    let ((wk_rx, wk_tx), (sv_rx, sv_tx)) = channel_pair();
    let ((sv_orch_rx, sv_orch_tx), (orch_sv_rx, orch_sv_tx)) = channel_pair();
    let ((wk_orch_rx, wk_orch_tx), (orch_wk_rx, orch_wk_tx)) = channel_pair();

    const MAX_EPOCHS: usize = 100;
    const NPARAMS: usize = 2;

    let shard_size = NonZeroUsize::new(1).unwrap();
    let mut param_gen = ConstParamGen::new(0.5, NPARAMS);
    let optimizer_factory = |_| GradientDescent::new(FloatPositive::new(0.1).unwrap());
    let store = BlockingStore::new(shard_size, &mut param_gen, optimizer_factory);
    let handle = StoreHandle::new(store);
    let synchronizer = BarrierSync::new(1);
    let transport = comms::build_simple_transport(sv_orch_rx, sv_orch_tx);
    let orch_handle = OrchHandle::new(transport);
    let mut server = ParameterServer::new(handle, synchronizer, orch_handle);

    let transport = comms::build_simple_transport(sv_rx, sv_tx);
    let worker_handle = WorkerHandle::new(0, transport);
    server.spawn(worker_handle);

    let worker_fut = mock_lineal_worker(wk_rx, wk_tx, wk_orch_rx, wk_orch_tx, MAX_EPOCHS, NPARAMS);
    let orchestrator_fut = mock_orch(orch_wk_rx, orch_wk_tx, orch_sv_rx, orch_sv_tx);
    let (_, _, params) = tokio::try_join!(worker_fut, server.run(), orchestrator_fut)?;
    println!("params: {params:?}");
    Ok(())
}
