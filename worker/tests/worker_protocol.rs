use comms::{OnoReceiver, OnoSender};
use machine_learning::middleware::ParamManager;
use tokio::io::{self, DuplexStream, ReadHalf, WriteHalf};

use comms::msg::{Command, Msg, Payload};
use machine_learning::training::{TrainResult, Trainer};
use worker::{middleware::Middleware, worker::Worker};

struct TestTrainer {
    buf: Vec<f32>,
    epoch: usize,
    max_epochs: usize,
    nparams: usize,
}

impl TestTrainer {
    fn new(max_epochs: usize, nparams: usize) -> Self {
        Self {
            buf: Vec::new(),
            nparams,
            max_epochs,
            epoch: 0,
        }
    }
}

impl Trainer for TestTrainer {
    fn train(&mut self, param_manager: &mut ParamManager<'_>) -> TrainResult {
        self.buf.clear();

        let mut front = param_manager.front();
        let params = front.next(self.nparams).unwrap();

        self.buf.extend(params.iter().map(|x| 2.0 * x));

        self.epoch += 1;
        TrainResult {
            losses: vec![0.0],
            was_last: self.epoch == self.max_epochs,
        }
    }
}

// server_addr: "127.0.0.1:0".parse().unwrap(),

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

#[tokio::test]
async fn worker_sends_gradient_on_weights() -> io::Result<()> {
    let ((mut sv_rx, mut sv_tx), (wk_rx, wk_tx)) = channel_pair();
    let ((orch_wk_rx, orch_wk_tx), _) = channel_pair();

    let worker = Worker::new(Box::new(TestTrainer::new(1, 2)));
    let mut middleware = Middleware::new(vec![0]);
    middleware.spawn(wk_rx, wk_tx, 2);

    let worker_fut = async move { worker.run(orch_wk_rx, orch_wk_tx, middleware).await };
    let server_fut = async move {
        let mut params = [1.0, 2.0];
        let msg = Msg::Data(Payload::Params(&mut params));
        sv_tx.send(&msg).await?;

        let mut rx_buf = vec![0; 128];
        let msg: Msg = sv_rx.recv_into(&mut rx_buf).await?;
        let Msg::Data(Payload::Grad(grad)) = msg else {
            panic!("unexpected message");
        };

        assert_eq!(grad, &[2.0, 4.0]);
        sv_tx.send(&Msg::Control(Command::Disconnect)).await?;
        Ok(())
    };

    tokio::try_join!(worker_fut, server_fut)?;
    Ok(())
}

#[tokio::test]
async fn worker_rejects_unexpected_message() -> io::Result<()> {
    let ((_, mut sv_tx), (wk_rx, wk_tx)) = channel_pair();
    let ((orch_wk_rx, orch_wk_tx), _) = channel_pair();

    let worker = Worker::new(Box::new(TestTrainer::new(1, 2)));
    let mut middleware = Middleware::new(vec![0]);
    middleware.spawn(wk_rx, wk_tx, 2);

    let worker_fut = async move { worker.run(orch_wk_rx, orch_wk_tx, middleware).await };
    let server_fut = async move {
        let grad = [0.1_f32, 0.2];
        let msg = Msg::Data(Payload::Grad(&grad));
        sv_tx.send(&msg).await
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

    let worker = Worker::new(Box::new(TestTrainer::new(1, 2)));
    let mut middleware = Middleware::new(vec![0]);
    middleware.spawn(wk_rx, wk_tx, 2);

    let worker_fut = async move { worker.run(orch_wk_rx, orch_wk_tx, middleware).await };
    let server_fut = async move {
        let msg = Msg::Control(Command::Disconnect);
        sv_tx.send(&msg).await
    };

    tokio::try_join!(worker_fut, server_fut)?;
    Ok(())
}

// TODO: Este lo deje armado para que funcione, de momento deje comentado el reporte de losses, por eso esta comentado el test
//
// #[tokio::test(flavor = "current_thread")]
// async fn worker_reports_losses_each_epoch_when_enabled() -> io::Result<()> {
//     let ((mut sv_rx, mut sv_tx), (wk_rx, wk_tx)) = channel_pair();
//     let ((orch_wk_rx, orch_wk_tx), (mut orch_rx, _orch_tx)) = channel_pair();

//     let worker = Worker::new(Box::new(TestTrainer::new(1, 2)));
//     let mut middleware = Middleware::new(vec![0]);
//     middleware.spawn(wk_rx, wk_tx, 2);

//     let worker_fut = async move { worker.run(orch_wk_rx, orch_wk_tx, middleware).await };
//     let server_fut = async move {
//         let mut params = [1.0, 2.0];
//         let msg = Msg::Data(Payload::Params(&mut params));
//         sv_tx.send(&msg).await?;

//         let mut rx_buf = vec![0; 128];
//         match sv_rx.recv_into(&mut rx_buf).await? {
//             Msg::Data(Payload::Grad(g)) => assert_eq!(g, &[2.0, 4.0]),
//             other => panic!("unexpected message: {other:?}"),
//         }

//         sv_tx.send(&Msg::Control(Command::Disconnect)).await?;
//         Ok(())
//     };

//     let orchestrator_fut = async move {
//         let mut rx_buf = vec![0; 128];
//         let msg: Msg = orch_rx.recv_into(&mut rx_buf).await?;
//         let res = matches!(msg, Msg::Control(Command::ReportLoss { losses: _ }));
//         Ok(res)
//     };

//     let (_, _, res) = tokio::try_join!(worker_fut, server_fut, orchestrator_fut)?;
//     assert!(res);
//     Ok(())
// }
