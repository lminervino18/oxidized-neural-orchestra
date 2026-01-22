use std::io;

use comms::{
    msg::{Command, Msg},
    specs::worker::WorkerSpec,
    OnoReceiver,
};
use log::{info, warn};
use tokio::io::AsyncRead;

/// Worker bootstrap acceptor.
pub struct WorkerAcceptor;

impl WorkerAcceptor {
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
    ///
    /// # Panics
    /// Never panics.
    pub async fn handshake<R>(rx: &mut OnoReceiver<R>) -> io::Result<Option<WorkerSpec>>
    where
        R: AsyncRead + Unpin + Send,
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
}
