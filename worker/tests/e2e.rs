use std::{
    io,
    net::{IpAddr, Ipv4Addr, SocketAddr},
    num::NonZeroUsize,
};

use tokio::io as tokio_io;

use comms::msg::{Command, Msg, Payload};
use comms::specs::server::OptimizerSpec;
use comms::specs::worker::{AlgorithmSpec, ModelSpec, TrainingSpec, WorkerSpec};

fn mk_spec(steps: usize, num_params: usize, offline_steps: usize) -> WorkerSpec {
    let ps_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 8765);

    WorkerSpec {
        worker_id: 0,
        steps: NonZeroUsize::new(steps).unwrap(),
        num_params: NonZeroUsize::new(num_params).unwrap(),
        model: ModelSpec::Mock,
        training: TrainingSpec {
            algorithm: AlgorithmSpec::ParameterServer { server_ip: ps_addr },
            optimizer: OptimizerSpec::GradientDescent { learning_rate: 0.1 },
            offline_steps,
            epochs: NonZeroUsize::new(1).unwrap(),
        },
        seed: None,
    }
}

async fn run_e2e(offline_steps: usize) -> io::Result<()> {
    const BUF_SIZE: usize = 4096;
    const STEPS: usize = 3;
    const PARAMS: usize = 2;

    let (sv_stream, wk_stream) = tokio_io::duplex(BUF_SIZE);

    let (sv_rx, sv_tx) = tokio_io::split(sv_stream);
    let (mut sv_rx, mut sv_tx) = comms::channel(sv_rx, sv_tx);

    let (wk_rx, wk_tx) = tokio_io::split(wk_stream);
    let (mut wk_rx, wk_tx) = comms::channel(wk_rx, wk_tx);

    let spec = mk_spec(STEPS, PARAMS, offline_steps);
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

    for step in 0..STEPS {
        let mut w = [step as f32 + 1.0, step as f32 + 2.0];
        sv_tx.send(&Msg::Data(Payload::Weights(&mut w))).await?;

        let msg: Msg = sv_rx.recv().await?;
        match msg {
            Msg::Data(Payload::Gradient(g)) => {
                assert_eq!(g.len(), PARAMS);
                assert_eq!(g[0], 2.0 * w[0]);
                assert_eq!(g[1], 2.0 * w[1]);
            }
            other => panic!("unexpected msg: {other:?}"),
        }
    }

    worker_task.await.unwrap()?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn worker_e2e_sends_expected_gradient_offline_0() -> io::Result<()> {
    run_e2e(0).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn worker_e2e_sends_expected_gradient_offline_nonzero() -> io::Result<()> {
    run_e2e(3).await
}
