use std::{
    io,
    net::{IpAddr, Ipv4Addr, SocketAddr},
    num::NonZeroUsize,
    time::Duration,
};

use machine_learning::MlError;
use tokio::{io as tokio_io, time::timeout};

use comms::msg::{Command, Msg, Payload};
use comms::specs::server::OptimizerSpec;
use comms::specs::worker::{AlgorithmSpec, ModelSpec, TrainingSpec, WorkerSpec};

async fn assert_no_gradient_received<R: tokio::io::AsyncRead + Unpin>(sv_rx: &mut comms::OnoReceiver<R>) {
    let recv_res = timeout(Duration::from_millis(50), sv_rx.recv::<Msg>()).await;

    match recv_res {
        Err(_) => {}
        Ok(Err(_)) => {}
        Ok(Ok(Msg::Data(Payload::Gradient(_)))) => {
            panic!("server unexpectedly received a Gradient message");
        }
        Ok(Ok(_)) => {}
    }
}

fn mk_spec(steps: usize, num_params: usize) -> WorkerSpec {
    let ps_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 8765);

    WorkerSpec {
        worker_id: 0,
        steps: NonZeroUsize::new(steps).unwrap(),
        num_params: NonZeroUsize::new(num_params).unwrap(),
        model: ModelSpec::Noop,
        training: TrainingSpec {
            algorithm: AlgorithmSpec::ParameterServer { server_ip: ps_addr },
            optimizer: OptimizerSpec::GradientDescent { learning_rate: 0.1 },
            offline_steps: 0,
            epochs: NonZeroUsize::new(1).unwrap(),
        },
        seed: None,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn worker_rejects_weight_length_change_across_steps() -> io::Result<()> {
    let (sv_stream, wk_stream) = tokio_io::duplex(4096);

    let (sv_rx, sv_tx) = tokio_io::split(sv_stream);
    let (mut sv_rx, mut sv_tx) = comms::channel(sv_rx, sv_tx);

    let (wk_rx, wk_tx) = tokio_io::split(wk_stream);
    let (mut wk_rx, wk_tx) = comms::channel(wk_rx, wk_tx);

    let spec = mk_spec(2, 2);
    sv_tx.send(&Msg::Control(Command::CreateWorker(spec))).await?;

    let worker_task = tokio::spawn(async move {
        let Some(spec) = worker::WorkerAcceptor::bootstrap(&mut wk_rx).await? else {
            return Ok(());
        };

        let worker = worker::WorkerBuilder::build(&spec);
        worker
            .run(wk_rx, wk_tx)
            .await
            .map_err(worker::WorkerError::into_io)
    });

    let mut w1 = [1.0_f32, 2.0];
    sv_tx.send(&Msg::Data(Payload::Weights(&mut w1))).await?;

    let msg: Msg = sv_rx.recv().await?;
    match msg {
        Msg::Data(Payload::Gradient(g)) => assert_eq!(g.len(), 2),
        other => panic!("unexpected msg: {other:?}"),
    }

    let mut w2 = [1.0_f32, 2.0, 3.0];
    sv_tx.send(&Msg::Data(Payload::Weights(&mut w2))).await?;

    let res = worker_task.await.unwrap();
    assert!(res.is_err());

    assert_no_gradient_received(&mut sv_rx).await;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn worker_rejects_unexpected_message() -> io::Result<()> {
    let (sv_stream, wk_stream) = tokio_io::duplex(4096);

    let (sv_rx, sv_tx) = tokio_io::split(sv_stream);
    let (mut sv_rx, mut sv_tx) = comms::channel(sv_rx, sv_tx);

    let (wk_rx, wk_tx) = tokio_io::split(wk_stream);
    let (mut wk_rx, wk_tx) = comms::channel(wk_rx, wk_tx);

    let spec = mk_spec(1, 2);
    sv_tx.send(&Msg::Control(Command::CreateWorker(spec))).await?;

    let worker_task = tokio::spawn(async move {
        let Some(spec) = worker::WorkerAcceptor::bootstrap(&mut wk_rx).await? else {
            return Ok(());
        };

        let worker = worker::WorkerBuilder::build(&spec);
        worker
            .run(wk_rx, wk_tx)
            .await
            .map_err(worker::WorkerError::into_io)
    });

    let g = [0.1_f32, 0.2];
    sv_tx.send(&Msg::Data(Payload::Gradient(&g))).await?;

    let res = worker_task.await.unwrap();
    assert!(res.is_err());

    assert_no_gradient_received(&mut sv_rx).await;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn acceptor_returns_spec_on_create_worker() -> io::Result<()> {
    let (sv_stream, wk_stream) = tokio_io::duplex(4096);

    let (sv_rx, sv_tx) = tokio_io::split(sv_stream);
    let (_sv_rx, mut sv_tx) = comms::channel(sv_rx, sv_tx);

    let (wk_rx, wk_tx) = tokio_io::split(wk_stream);
    let (mut wk_rx, _wk_tx) = comms::channel(wk_rx, wk_tx);

    let spec = mk_spec(3, 4);
    sv_tx.send(&Msg::Control(Command::CreateWorker(spec))).await?;

    let got = worker::WorkerAcceptor::bootstrap(&mut wk_rx).await?;
    assert!(got.is_some());

    let got = got.unwrap();
    assert_eq!(got.worker_id, 0);
    assert_eq!(got.steps.get(), 3);
    assert_eq!(got.num_params.get(), 4);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn acceptor_returns_none_on_disconnect_before_bootstrap() -> io::Result<()> {
    let (sv_stream, wk_stream) = tokio_io::duplex(4096);

    let (sv_rx, sv_tx) = tokio_io::split(sv_stream);
    let (_sv_rx, mut sv_tx) = comms::channel(sv_rx, sv_tx);

    let (wk_rx, wk_tx) = tokio_io::split(wk_stream);
    let (mut wk_rx, _wk_tx) = comms::channel(wk_rx, wk_tx);

    sv_tx.send(&Msg::Control(Command::Disconnect)).await?;

    let got = worker::WorkerAcceptor::bootstrap(&mut wk_rx).await?;
    assert!(got.is_none());
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn acceptor_ignores_noise_until_create_worker() -> io::Result<()> {
    let (sv_stream, wk_stream) = tokio_io::duplex(4096);

    let (sv_rx, sv_tx) = tokio_io::split(sv_stream);
    let (_sv_rx, mut sv_tx) = comms::channel(sv_rx, sv_tx);

    let (wk_rx, wk_tx) = tokio_io::split(wk_stream);
    let (mut wk_rx, _wk_tx) = comms::channel(wk_rx, wk_tx);

    let mut w = [1.0_f32, 2.0];
    sv_tx.send(&Msg::Data(Payload::Weights(&mut w))).await?;

    let spec = mk_spec(1, 2);
    sv_tx.send(&Msg::Control(Command::CreateWorker(spec))).await?;

    let got = worker::WorkerAcceptor::bootstrap(&mut wk_rx).await?;
    assert!(got.is_some());
    Ok(())
}

#[test]
fn worker_error_into_io_maps_kinds() {
    let err = worker::WorkerError::UnexpectedMessage { step: 0, got: "control" };
    let io_err = err.into_io();
    assert_eq!(io_err.kind(), io::ErrorKind::InvalidData);

    let err = worker::WorkerError::WeightsLengthMismatch { step: 0, got: 1, expected: 2 };
    let io_err = err.into_io();
    assert_eq!(io_err.kind(), io::ErrorKind::InvalidData);

    let err = worker::WorkerError::ComputeFailed {
        step: 0,
        source: MlError::ShapeMismatch { what: "params", got: 1, expected: 2 },
    };
    let io_err = err.into_io();
    assert_eq!(io_err.kind(), io::ErrorKind::InvalidData);

    let src = io::Error::new(io::ErrorKind::TimedOut, "x");
    let err = worker::WorkerError::Io(src);
    let io_err = err.into_io();
    assert_eq!(io_err.kind(), io::ErrorKind::TimedOut);
}
