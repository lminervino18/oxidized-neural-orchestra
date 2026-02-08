use std::io;

use tokio::io as tokio_io;

use comms::msg::{Command, Msg, Payload};
use comms::specs::worker::AlgorithmSpec;
use machine_learning::training::Trainer;
use worker::{Worker, WorkerError};

struct TestTrainer {
    buf: Vec<f32>,
}

impl TestTrainer {
    fn new() -> Self {
        Self { buf: Vec::new() }
    }
}

impl Trainer for TestTrainer {
    fn train(&mut self, weights: &mut [f32]) -> &[f32] {
        self.buf.clear();
        self.buf.extend(weights.iter().map(|x| 2.0 * x));
        &self.buf
    }
}

fn make_worker() -> Worker {
    Worker::new(
        0,
        AlgorithmSpec::ParameterServer {
            server_ip: "127.0.0.1:0".parse().unwrap(),
        },
        Box::new(TestTrainer::new()),
    )
}

#[tokio::test(flavor = "current_thread")]
async fn worker_sends_gradient_on_weights() -> io::Result<()> {
    let (sv_stream, wk_stream) = tokio_io::duplex(4096);

    let (sv_rx, sv_tx) = tokio_io::split(sv_stream);
    let (mut sv_rx, mut sv_tx) = comms::channel(sv_rx, sv_tx);

    let (wk_rx, wk_tx) = tokio_io::split(wk_stream);
    let (wk_rx, wk_tx) = comms::channel(wk_rx, wk_tx);

    let worker = make_worker();

    let worker_fut = async move {
        worker
            .run_parameter_server(wk_rx, wk_tx)
            .await
            .map_err(WorkerError::into_io)
    };

    let server_fut = async move {
        let mut w = [1.0_f32, 2.0];
        sv_tx.send(&Msg::Data(Payload::Params(&mut w))).await?;

        let msg: Msg = sv_rx.recv().await?;
        match msg {
            Msg::Data(Payload::Grad(g)) => assert_eq!(g, &[2.0, 4.0]),
            other => panic!("unexpected message: {other:?}"),
        }

        sv_tx.send(&Msg::Control(Command::Disconnect)).await?;
        Ok::<(), io::Error>(())
    };

    let (worker_res, server_res) = tokio::join!(worker_fut, server_fut);
    server_res?;
    worker_res?;
    Ok(())
}

#[tokio::test(flavor = "current_thread")]
async fn worker_rejects_unexpected_message() -> io::Result<()> {
    let (sv_stream, wk_stream) = tokio_io::duplex(4096);

    let (sv_rx, sv_tx) = tokio_io::split(sv_stream);
    let (_sv_rx, mut sv_tx) = comms::channel(sv_rx, sv_tx);

    let (wk_rx, wk_tx) = tokio_io::split(wk_stream);
    let (wk_rx, wk_tx) = comms::channel(wk_rx, wk_tx);

    let worker = make_worker();

    let worker_fut = async move {
        worker
            .run_parameter_server(wk_rx, wk_tx)
            .await
            .map_err(WorkerError::into_io)
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
    let (sv_stream, wk_stream) = tokio_io::duplex(4096);

    let (sv_rx, sv_tx) = tokio_io::split(sv_stream);
    let (_sv_rx, mut sv_tx) = comms::channel(sv_rx, sv_tx);

    let (wk_rx, wk_tx) = tokio_io::split(wk_stream);
    let (wk_rx, wk_tx) = comms::channel(wk_rx, wk_tx);

    let worker = make_worker();

    let worker_fut = async move {
        worker
            .run_parameter_server(wk_rx, wk_tx)
            .await
            .map_err(WorkerError::into_io)
    };

    let server_fut = async move {
        sv_tx.send(&Msg::Control(Command::Disconnect)).await?;
        Ok::<(), io::Error>(())
    };

    let (worker_res, server_res) = tokio::join!(worker_fut, server_fut);
    server_res?;
    worker_res?;
    Ok(())
}
