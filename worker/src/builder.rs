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
}
