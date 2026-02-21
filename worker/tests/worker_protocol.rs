use std::num::NonZeroUsize;

use comms::{OnoReceiver, OnoSender};
use tokio::io::{self, DuplexStream, ReadHalf, WriteHalf};

use comms::msg::{Command, Msg, Payload};
use comms::specs::worker::AlgorithmSpec;
use machine_learning::training::Trainer;
use worker::Worker;

struct TestTrainer {
    buf: Vec<f32>,
}

impl TestTrainer {
    fn new() -> Self {
        Self { buf: Vec::new() }
    }
}

impl Trainer for TestTrainer {
    fn train(&mut self, weights: &mut [f32]) -> (&[f32], Vec<f32>) {
        self.buf.clear();
        self.buf.extend(weights.iter().map(|x| 2.0 * x));
        (&self.buf, vec![0.0])
    }
}

fn make_worker() -> Worker {
    Worker::new(
        0,
        NonZeroUsize::new(1).unwrap(),
        AlgorithmSpec::ParameterServer {
            server_addr: "127.0.0.1:0".parse().unwrap(),
        },
        Box::new(TestTrainer::new()),
    )
}

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

#[tokio::test(flavor = "current_thread")]
async fn worker_sends_gradient_on_weights() -> io::Result<()> {
    let ((mut sv_rx, mut sv_tx), (wk_rx, wk_tx)) = channel_pair();
    let ((orch_wk_rx, orch_wk_tx), _) = channel_pair();
    let worker = make_worker();

    let worker_fut = async move {
        worker
            .run_parameter_server(wk_rx, wk_tx, orch_wk_rx, orch_wk_tx)
            .await
            .map_err(io::Error::from)
    };

    let server_fut = async move {
        let mut w = [1.0_f32, 2.0];
        sv_tx.send(&Msg::Data(Payload::Params(&mut w))).await?;

        let mut buf = vec![0; 128];
        let msg: Msg = sv_rx.recv_into(&mut buf).await?;
        match msg {
            Msg::Data(Payload::Grad(g)) => assert_eq!(g, &[2.0, 4.0]),
            other => panic!("unexpected message: {other:?}"),
        }

        sv_tx.send(&Msg::Control(Command::Disconnect)).await?;
        Ok::<(), io::Error>(())
    };

    tokio::try_join!(worker_fut, server_fut)?;
    Ok(())
}

#[tokio::test(flavor = "current_thread")]
async fn worker_rejects_unexpected_message() -> io::Result<()> {
    let ((_, mut sv_tx), (wk_rx, wk_tx)) = channel_pair();
    let ((orch_wk_rx, orch_wk_tx), _) = channel_pair();
    let worker = make_worker();

    let worker_fut = async move {
        worker
            .run_parameter_server(wk_rx, wk_tx, orch_wk_rx, orch_wk_tx)
            .await
            .map_err(io::Error::from)
    };

    let server_fut = async move {
        let g = [0.1_f32, 0.2];
        sv_tx.send(&Msg::Data(Payload::Grad(&g))).await?;
        Ok::<(), io::Error>(())
    };

    let (worker_res, server_res) = tokio::join!(worker_fut, server_fut);
    server_res?;
    assert!(worker_res.is_err());
    Ok(())
}

#[tokio::test(flavor = "current_thread")]
async fn worker_stops_on_disconnect() -> io::Result<()> {
    let ((_, mut sv_tx), (wk_rx, wk_tx)) = channel_pair();
    let ((orch_wk_rx, orch_wk_tx), _) = channel_pair();
    let worker = make_worker();

    let worker_fut = async move {
        worker
            .run_parameter_server(wk_rx, wk_tx, orch_wk_rx, orch_wk_tx)
            .await
            .map_err(io::Error::from)
    };

    let server_fut = async move {
        sv_tx.send(&Msg::Control(Command::Disconnect)).await?;
        Ok::<(), io::Error>(())
    };

    tokio::try_join!(worker_fut, server_fut)?;
    Ok(())
}

#[tokio::test(flavor = "current_thread")]
async fn worker_reports_losses_each_epoch_when_enabled() -> io::Result<()> {
    let ((mut sv_rx, mut sv_tx), (wk_rx, wk_tx)) = channel_pair();
    let ((orch_wk_rx, orch_wk_tx), (mut orch_rx, _orch_tx)) = channel_pair();
    let worker = make_worker();

    let worker_fut = async move {
        worker
            .run_parameter_server(wk_rx, wk_tx, orch_wk_rx, orch_wk_tx)
            .await
            .map_err(io::Error::from)
    };

    let server_fut = async move {
        let mut w = [1.0_f32, 2.0];
        sv_tx.send(&Msg::Data(Payload::Params(&mut w))).await?;

        let mut buf = vec![0; 128];
        let msg: Msg = sv_rx.recv_into(&mut buf).await?;
        match msg {
            Msg::Data(Payload::Grad(g)) => assert_eq!(g, &[2.0, 4.0]),
            other => panic!("unexpected message: {other:?}"),
        }

        sv_tx.send(&Msg::Control(Command::Disconnect)).await?;
        Ok::<(), io::Error>(())
    };

    let orchestrator_fut = async move {
        let mut buf = vec![0; 128];
        let msg: Msg = orch_rx.recv_into(&mut buf).await?;
        match msg {
            Msg::Control(Command::ReportLoss {
                worker_id,
                losses: _,
            }) => {
                assert_eq!(worker_id, 0);
            }
            other => panic!("unexpected orchestrator message: {other:?}"),
        }
        Ok::<(), io::Error>(())
    };

    tokio::try_join!(worker_fut, server_fut, orchestrator_fut)?;
    Ok(())
}
