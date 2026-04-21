use std::{env, num::NonZeroUsize};

use machine_learning::{
    arch::{Sequential, layers::Layer, loss::Mse},
    dataset::{Dataset, DatasetSrc},
    optimization::{GradientDescent, Optimizer},
    training::BackpropTrainer,
};
use tokio::io::{self, AsyncRead, AsyncWrite, DuplexStream, ReadHalf, WriteHalf};

use comms::{
    OrchEvent, OrchHandle, ParamServerHandle, PullParamsResponse, Stp, WorkerEvent, WorkerHandle,
};
use worker::{cluster_manager::ServerClusterManager, worker::ParameterServerWorker};

#[allow(clippy::type_complexity)]
fn channel_pair() -> (
    (ReadHalf<DuplexStream>, WriteHalf<DuplexStream>),
    (ReadHalf<DuplexStream>, WriteHalf<DuplexStream>),
) {
    let (stream1, stream2) = io::duplex(4096);
    let rxtx1 = io::split(stream1);
    let rxtx2 = io::split(stream2);
    (rxtx1, rxtx2)
}

async fn mock_orch<R, W>(
    mut worker_handle: WorkerHandle<Stp<R, W>>,
    mut server_handle: ParamServerHandle<Stp<R, W>>,
) -> io::Result<Vec<f32>>
where
    R: AsyncRead + Unpin + Send,
    W: AsyncWrite + Unpin + Send,
{
    loop {
        match worker_handle.recv_event().await? {
            WorkerEvent::Disconnect => break,
            WorkerEvent::Loss(losses) => println!("loss: {losses:?}"),
            _ => {}
        }
    }

    let PullParamsResponse::Params(params) = server_handle.pull_params().await?;
    let params = params.to_vec();
    server_handle.disconnect().await?;
    Ok(params)
}

async fn mock_server<R, W>(
    mut worker_handle: WorkerHandle<Stp<R, W>>,
    mut orch_handle: OrchHandle<Stp<R, W>>,
    learning_rate: f32,
    nparams: usize,
) -> io::Result<()>
where
    R: AsyncRead + Unpin + Send,
    W: AsyncWrite + Unpin + Send,
{
    let mut optimizer = GradientDescent::new(learning_rate);
    let mut params = vec![0.5; nparams];

    loop {
        match worker_handle.recv_event().await? {
            WorkerEvent::Disconnect => break,
            WorkerEvent::Grad(grad) => {
                optimizer.update_params(grad, &mut params).unwrap();
            }
            WorkerEvent::RequestParams => {
                worker_handle.push_params(&mut params).await?;
            }
            _ => {}
        }
    }

    loop {
        match orch_handle.recv_event().await? {
            OrchEvent::Disconnect => break,
            OrchEvent::RequestParams => {
                orch_handle.push_params(&mut params).await?;
            }
            _ => unreachable!(),
        }
    }

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_local_lineal_model_convergence() -> io::Result<()> {
    unsafe { env::set_var("RUST_BACKTRACE", "full") };

    let ((sv_rx, sv_tx), (wk_rx, wk_tx)) = channel_pair();
    let ((wk_orch_rx, wk_orch_tx), (orch_wk_rx, orch_wk_tx)) = channel_pair();
    let ((sv_orch_rx, sv_orch_tx), (orch_sv_rx, orch_sv_tx)) = channel_pair();

    const MAX_EPOCHS: usize = 100;

    let model = Sequential::new(vec![Layer::dense((1, 1))]);
    let x_size = NonZeroUsize::new(1).unwrap();
    let y_size = NonZeroUsize::new(1).unwrap();
    let trainer = BackpropTrainer::new(
        model,
        vec![GradientDescent::new(0.1)],
        Dataset::new(
            DatasetSrc::inmem(vec![0., 1., 2., 3.], vec![1., 2., 3., 4.]),
            x_size,
            y_size,
        ),
        Mse::new(),
        0,
        NonZeroUsize::new(MAX_EPOCHS).unwrap(),
        NonZeroUsize::new(4).unwrap(),
        rand::rng(),
    );

    // Worker node.
    let transport = Stp::new(wk_orch_rx, wk_orch_tx);
    let orch_wk_handle = OrchHandle::new(transport);
    let worker = ParameterServerWorker::new(Box::new(trainer));
    let transport = Stp::new(sv_rx, sv_tx);
    let server_wk_handle = ParamServerHandle::new(0, transport);
    let mut cluster_manager = ServerClusterManager::new(vec![0]);
    cluster_manager.spawn(server_wk_handle, 2);

    // Server node.
    let transport = Stp::new(wk_rx, wk_tx);
    let worker_sv_handle = WorkerHandle::new(0, transport);
    let transport = Stp::new(sv_orch_rx, sv_orch_tx);
    let orch_sv_handle = OrchHandle::new(transport);

    // Orch node.
    let transport = Stp::new(orch_wk_rx, orch_wk_tx);
    let worker_orch_handle = WorkerHandle::new(0, transport);
    let transport = Stp::new(orch_sv_rx, orch_sv_tx);
    let server_orch_handle = ParamServerHandle::new(0, transport);

    let worker_fut = worker.run(orch_wk_handle, cluster_manager);
    let server_fut = mock_server(worker_sv_handle, orch_sv_handle, 0.1, 2);
    let orch_fut = mock_orch(worker_orch_handle, server_orch_handle);

    let (wk, sv, orch) = tokio::join!(worker_fut, server_fut, orch_fut);

    println!("worker: {wk:?}");
    println!("server: {sv:?}");
    println!("orchestrator: {orch:?}");

    let params = orch.unwrap();
    println!("params: {params:?}");

    Ok(())
}
