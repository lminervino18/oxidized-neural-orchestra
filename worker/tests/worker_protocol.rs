use comms::{OnoReceiver, OnoSender};
use machine_learning::middleware::ParamManager;
use tokio::io::{self, AsyncRead, DuplexStream, ReadHalf, WriteHalf};

use comms::msg::{Command, Msg, Payload};
use machine_learning::{
    Result,
    training::{TrainResult, Trainer},
};
use worker::{middleware::Middleware, worker::Worker};

struct TestTrainer {
    nparams: usize,
    max_epochs: usize,
    epoch: usize,
    losses: Vec<f32>,
}

impl TestTrainer {
    fn new(max_epochs: usize, nparams: usize) -> Self {
        Self {
            max_epochs,
            epoch: 0,
            nparams,
            losses: vec![0.0],
        }
    }
}

impl Trainer for TestTrainer {
    fn train(&mut self, param_manager: &mut ParamManager<'_>) -> Result<TrainResult<'_>> {
        self.epoch += 1;

        param_manager.zero_grad();
        let mut back = param_manager.back();
        let (params, grad) = back.next(self.nparams).unwrap();

        for (p, g) in params.iter().zip(grad) {
            *g = 2.0 * *p;
        }

        let res = TrainResult {
            losses: &self.losses,
            was_last: self.epoch == self.max_epochs,
        };

        Ok(res)
    }
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

async fn mock_orch<R: AsyncRead + Unpin>(mut rx: OnoReceiver<R>) -> io::Result<()> {
    let mut rx_buf = vec![0; 128];
    loop {
        match rx.recv_into(&mut rx_buf).await? {
            Msg::Control(Command::Disconnect) => break,
            _ => {}
        }
    }

    Ok::<_, io::Error>(())
}

#[tokio::test]
async fn worker_sends_gradient_on_weights() -> io::Result<()> {
    let ((mut sv_rx, mut sv_tx), (wk_rx, wk_tx)) = channel_pair();
    let ((orch_wk_rx, orch_wk_tx), (orch_rx, _)) = channel_pair();

    let worker = Worker::new(Box::new(TestTrainer::new(1, 2)));
    let mut middleware = Middleware::new(vec![0]);
    middleware.spawn(wk_rx, wk_tx, 2);

    let worker_fut = async move { worker.run(orch_wk_rx, orch_wk_tx, middleware).await };
    let server_fut = async move {
        let mut params = [1.0, 2.0];
        let msg = Msg::Data(Payload::Params(&mut params));
        sv_tx.send(&msg).await?;

        let mut rx_buf = vec![0; 128];
        let Msg::Data(Payload::Grad(grad)) = sv_rx.recv_into(&mut rx_buf).await? else {
            panic!("unexpected message");
        };

        let grad = grad.to_vec();
        let Msg::Control(Command::Disconnect) = sv_rx.recv_into(&mut rx_buf).await? else {
            panic!("unexpected message");
        };

        sv_tx.send(&Msg::Control(Command::Disconnect)).await?;
        assert_eq!(grad, &[2.0, 4.0]);
        Ok(())
    };

    tokio::try_join!(worker_fut, server_fut, mock_orch(orch_rx))?;
    Ok(())
}

#[tokio::test(flavor = "current_thread")]
async fn worker_reports_losses_each_epoch_when_enabled() -> io::Result<()> {
    let ((mut sv_rx, mut sv_tx), (wk_rx, wk_tx)) = channel_pair();
    let ((orch_wk_rx, orch_wk_tx), (mut orch_rx, _orch_tx)) = channel_pair();

    let worker = Worker::new(Box::new(TestTrainer::new(1, 2)));
    let mut middleware = Middleware::new(vec![0]);
    middleware.spawn(wk_rx, wk_tx, 2);

    let worker_fut = async move { worker.run(orch_wk_rx, orch_wk_tx, middleware).await };
    let server_fut = async move {
        let mut params = [1.0, 2.0];
        let msg = Msg::Data(Payload::Params(&mut params));
        sv_tx.send(&msg).await?;

        let mut rx_buf = vec![0; 128];
        let Msg::Data(Payload::Grad(grad)) = sv_rx.recv_into(&mut rx_buf).await? else {
            panic!("unexpected message");
        };

        let grad = grad.to_vec();
        let Msg::Control(Command::Disconnect) = sv_rx.recv_into(&mut rx_buf).await? else {
            panic!("unexpected message");
        };

        sv_tx.send(&Msg::Control(Command::Disconnect)).await?;
        assert_eq!(grad, &[2.0, 4.0]);
        Ok(())
    };

    let orchestrator_fut = async move {
        let mut rx_buf = vec![0; 128];
        let msg: Msg = orch_rx.recv_into(&mut rx_buf).await?;
        let res = matches!(msg, Msg::Control(Command::ReportLoss { losses: _ }));
        Ok(res)
    };

    let (_, _, res) = tokio::try_join!(worker_fut, server_fut, orchestrator_fut)?;
    assert!(res);
    Ok(())
}
