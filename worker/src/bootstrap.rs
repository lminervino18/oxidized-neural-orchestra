use std::io;

use comms::{
    msg::{Command, Msg},
    OnoReceiver, OnoSender,
};
use log::{debug, info, warn};
use ml_core::TrainStrategy;
use tokio::io::{AsyncRead, AsyncWrite};

use crate::{Worker, WorkerConfig};

/// Runs a worker after receiving a `CreateWorker(WorkerSpec)` control message.
///
/// This is the worker-side bootstrap handshake, analogous to the parameter server
/// waiting for `CreateServer(ServerSpec)`.
pub async fn run_bootstrapped<R, W, S>(mut rx: OnoReceiver<R>, tx: OnoSender<W>, strategy: S) -> io::Result<()>
where
    R: AsyncRead + Unpin + Send,
    W: AsyncWrite + Unpin + Send,
    S: TrainStrategy,
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
            Err(e) => {
                return Err(e);
            }
        }
    };

    debug!(
        worker_id = spec.worker_id,
        steps = spec.steps.get(),
        num_params = spec.num_params.get();
        "received worker spec"
    );

    let cfg = WorkerConfig::from_spec(&spec);
    let worker = Worker::new(cfg, spec.num_params, strategy);

    worker.run(rx, tx).await
}
