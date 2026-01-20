use std::io;

use comms::{
    msg::{Command, Msg},
    specs::worker::WorkerSpec,
    OnoReceiver,
};
use log::{debug, info, warn};
use ml_core::TrainStrategy;
use tokio::io::AsyncRead;

use crate::{Worker, WorkerConfig};

/// Worker bootstrap builder.
pub struct WorkerBuilder;

impl WorkerBuilder {
    /// Receives `CreateWorker(WorkerSpec)` and returns the spec.
    ///
    /// # Args
    /// * `rx` - Receiving end of the communication channel.
    ///
    /// # Returns
    /// Returns `Ok(Some(spec))` on `CreateWorker`.
    /// Returns `Ok(None)` if `Disconnect` is received before bootstrap.
    ///
    /// # Errors
    /// Returns `io::Error` if receiving fails.
    pub async fn handshake<R>(rx: &mut OnoReceiver<R>) -> io::Result<Option<WorkerSpec>>
    where
        R: AsyncRead + std::marker::Unpin + Send,
    {
        info!("waiting for CreateWorker spec");

        let spec = loop {
            match rx.recv::<Msg>().await {
                Ok(Msg::Control(Command::CreateWorker(spec))) => break spec,
                Ok(Msg::Control(Command::Disconnect)) => {
                    info!("received Disconnect before bootstrap, exiting");
                    return Ok(None);
                }
                Ok(msg) => warn!("expected CreateWorker, got {msg:?}"),
                Err(e) => return Err(e),
            }
        };

        debug!(
            worker_id = spec.worker_id,
            steps = spec.steps.get(),
            num_params = spec.num_params.get();
            "received worker spec"
        );

        Ok(Some(spec))
    }

    /// Builds a `Worker` from a `WorkerSpec`.
    ///
    /// # Args
    /// * `spec` - Bootstrap specification received from the orchestrator.
    /// * `strategy` - Concrete `TrainStrategy` implementation for this worker.
    ///
    /// # Returns
    /// A fully initialized `Worker` instance.
    pub fn build<S>(spec: &WorkerSpec, strategy: S) -> Worker<S>
    where
        S: TrainStrategy,
    {
        let cfg = WorkerConfig::from_spec(spec);
        Worker::new(cfg, spec.num_params, strategy)
    }

    /// Receives `CreateWorker(WorkerSpec)` and builds a `Worker`.
    ///
    /// # Args
    /// * `rx` - Receiving end of the communication channel.
    /// * `make_strategy` - Factory that builds the concrete `TrainStrategy` from the received spec.
    ///
    /// # Returns
    /// Returns `Ok(Some(worker))` on `CreateWorker`.
    /// Returns `Ok(None)` if `Disconnect` is received before bootstrap.
    ///
    /// # Errors
    /// Returns `io::Error` if receiving fails or the strategy factory fails.
    pub async fn from_handshake<R, S, F>(
        rx: &mut OnoReceiver<R>,
        make_strategy: F,
    ) -> io::Result<Option<Worker<S>>>
    where
        R: AsyncRead + std::marker::Unpin + Send,
        S: TrainStrategy,
        F: FnOnce(&WorkerSpec) -> io::Result<S>,
    {
        let Some(spec) = Self::handshake(rx).await? else {
            return Ok(None);
        };

        let strategy = make_strategy(&spec)?;
        Ok(Some(Self::build(&spec, strategy)))
    }
}
