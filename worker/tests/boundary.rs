use std::{io, num::NonZeroUsize, time::Duration};

use tokio::io as tokio_io;
use tokio::time::timeout;

use comms::msg::{Command, Msg, Payload};
use comms::specs::worker::{StrategySpec, WorkerSpec};

async fn assert_no_gradient_received<R: tokio::io::AsyncRead + Unpin>(
    sv_rx: &mut comms::OnoReceiver<R>,
) {
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
    WorkerSpec {
        worker_id: 0,
        steps: NonZeroUsize::new(steps).unwrap(),
        num_params: NonZeroUsize::new(num_params).unwrap(),
        strategy: StrategySpec::Noop,
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
    sv_tx
        .send(&Msg::Control(Command::CreateWorker(spec)))
        .await?;

    let worker_task = tokio::spawn(async move {
        let Some(spec) = worker::WorkerAcceptor::handshake(&mut wk_rx).await? else {
            return Ok(());
        };

        let worker = worker::WorkerBuilder::build(&spec)?;
        worker.run(wk_rx, wk_tx).await
    });

    let mut w1 = [1.0_f32, 2.0];
    sv_tx.send(&Msg::Data(Payload::Weights(&mut w1))).await?;

    // Step 0 completes; NoopStrategy still sends a gradient of the inferred length.
    let msg: Msg = sv_rx.recv().await?;
    match msg {
        Msg::Data(Payload::Gradient(g)) => assert_eq!(g.len(), 2),
        other => panic!("unexpected msg: {other:?}"),
    }

    // Step 1 uses a different weight length -> must error.
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
    sv_tx
        .send(&Msg::Control(Command::CreateWorker(spec)))
        .await?;

    let worker_task = tokio::spawn(async move {
        let Some(spec) = worker::WorkerAcceptor::handshake(&mut wk_rx).await? else {
            return Ok(());
        };

        let worker = worker::WorkerBuilder::build(&spec)?;
        worker.run(wk_rx, wk_tx).await
    });

    let g = [0.1_f32, 0.2];
    sv_tx.send(&Msg::Data(Payload::Gradient(&g))).await?;

    let res = worker_task.await.unwrap();
    assert!(res.is_err());

    assert_no_gradient_received(&mut sv_rx).await;
    Ok(())
}
