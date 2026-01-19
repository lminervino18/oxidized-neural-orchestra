use std::io;

use comms::{
    msg::{Command, Msg},
    specs::worker::WorkerSpec,
    OnoReceiver, OnoSender,
};
use log::{debug, info, warn};
use ml_core::TrainStrategy;
use tokio::io::{AsyncRead, AsyncWrite};

use crate::{Worker, WorkerConfig};

/// Runs a worker after receiving a `CreateWorker(WorkerSpec)` control message.
pub async fn run_bootstrapped<R, W, S, F>(
    mut rx: OnoReceiver<R>,
    tx: OnoSender<W>,
    make_strategy: F,
) -> io::Result<()>
where
    R: AsyncRead + Unpin + Send,
    W: AsyncWrite + Unpin + Send,
    S: TrainStrategy,
    F: FnOnce(&WorkerSpec) -> io::Result<S>,
{
    info!("waiting for CreateWorker spec");

    let spec = loop {
        match rx.recv::<Msg>().await {
            Ok(Msg::Control(Command::CreateWorker(spec))) => break spec,
            Ok(Msg::Control(Command::Disconnect)) => {
                info!("received Disconnect before bootstrap, exiting");
                return Ok(());
            }
            Ok(msg) => {
                warn!("expected CreateWorker, got {msg:?}");
            }
            Err(e) => return Err(e),
        }
    };

    debug!(
        worker_id = spec.worker_id,
        steps = spec.steps.get(),
        num_params = spec.num_params.get(),
        strategy_kind = spec.strategy.kind.as_str();
        "received worker spec"
    );

    let strategy = make_strategy(&spec)?;

    let cfg = WorkerConfig::from_spec(&spec);
    let worker = Worker::new(cfg, spec.num_params, strategy);

    worker.run(rx, tx).await
}
