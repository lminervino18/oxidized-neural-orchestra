use std::num::NonZeroUsize;

use comms::{OnoReceiver, OnoSender};
use machine_learning::{
    arch::{Sequential, layers::Layer, loss::Mse},
    dataset::Dataset,
    optimization::{GradientDescent, Optimizer},
    training::ModelTrainer,
};
use tokio::io::{self, AsyncRead, AsyncWrite, DuplexStream, ReadHalf, WriteHalf};

use comms::msg::{Command, Msg, Payload};
use worker::{middleware::Middleware, worker::Worker};

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

async fn mock_orch<R>(mut wk_rx: OnoReceiver<R>, mut sv_rx: OnoReceiver<R>) -> io::Result<Vec<f32>>
where
    R: AsyncRead + Unpin,
{
    let mut rx_buf = vec![0; 128];

    loop {
        match wk_rx.recv_into(&mut rx_buf).await? {
            Msg::Control(Command::Disconnect) => break,
            Msg::Control(Command::ReportLoss { losses }) => println!("loss: {losses:?}"),
            _ => {}
        }
    }

    let Msg::Data(Payload::Params(params)) = sv_rx.recv_into(&mut rx_buf).await? else {
        return Err(io::Error::other(
            "received an invalid message kind from the server",
        ));
    };

    Ok(params.to_vec())
}

async fn mock_server<R, W>(
    mut rx: OnoReceiver<R>,
    mut tx: OnoSender<W>,
    mut orch_tx: OnoSender<W>,
    nparams: usize,
) -> io::Result<()>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let mut optimizer = GradientDescent::new(1.0);
    let mut params = vec![0.5; nparams];
    let mut rx_buf = vec![0; 128];

    let msg = Msg::Data(Payload::Params(&mut params));
    tx.send(&msg).await?;

    loop {
        match rx.recv_into(&mut rx_buf).await? {
            Msg::Control(Command::Disconnect) => {
                break;
            }
            Msg::Data(Payload::Grad(grad)) => {
                optimizer.update_params(grad, &mut params).unwrap();
                let msg = Msg::Data(Payload::Params(&mut params));
                tx.send(&msg).await?;
            }
            _ => {}
        }
    }

    let msg = Msg::Control(Command::Disconnect);
    tx.send(&msg).await?;

    let msg = Msg::Data(Payload::Params(&mut params));
    orch_tx.send(&msg).await?;

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_local_lineal_model_convergence() -> io::Result<()> {
    let ((sv_rx, sv_tx), (wk_rx, wk_tx)) = channel_pair();
    let ((wk_orch_rx, wk_orch_tx), (orch_wk_rx, _)) = channel_pair();
    let ((_, sv_orch_tx), (orch_sv_rx, _)) = channel_pair();

    const MAX_EPOCHS: usize = 100;

    let model = Sequential::new([Layer::dense((1, 1), None)]);
    let trainer = ModelTrainer::new(
        model,
        vec![GradientDescent::new(0.1)],
        Dataset::new(vec![0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0], 1, 1),
        0,
        NonZeroUsize::new(MAX_EPOCHS).unwrap(),
        NonZeroUsize::new(4).unwrap(),
        Mse::new(),
        rand::rng(),
    );

    let worker = Worker::new(Box::new(trainer));
    let mut middleware = Middleware::new(vec![0]);
    middleware.spawn(wk_rx, wk_tx, 2);

    let worker_fut = worker.run(wk_orch_rx, wk_orch_tx, middleware);
    let server_fut = mock_server(sv_rx, sv_tx, sv_orch_tx, 2);
    let orch_fut = mock_orch(orch_wk_rx, orch_sv_rx);
    let (_, _, params) = tokio::try_join!(worker_fut, server_fut, orch_fut)?;
    println!("params: {params:?}");
    Ok(())
}
