use std::{io, time::Duration};

use tokio::time;

use super::TransportLayer;
use crate::protocol::Msg;

/// The `TimeOuter` tries receiving messages inside a time window.
/// If it fails it returns an error with `ErrorKind::TimedOut`.
pub struct TimeOuter<L: TransportLayer> {
    timeout: Duration,
    inner: L,
}

impl<L: TransportLayer> TimeOuter<L> {
    /// Creates a new `TimeOuter` transport layer.
    ///
    /// # Args
    /// * `timeout` - The duration to wait until receiving a message and declaring an error.
    /// * `inner` - The inner transport layer stack.
    ///
    /// # Returns
    /// A new `TimeOuter` transport layer instance.
    pub fn new(timeout: Duration, inner: L) -> Self {
        Self { timeout, inner }
    }
}

impl<L: TransportLayer> TransportLayer for TimeOuter<L> {
    /// Calls receive on the inner transport layer setting it's timeout.
    /// Returning an io error with `ErrorKind::TimedOut` if the timoeut
    /// reaches the setted duration.
    ///
    /// # Returns
    /// A deserialized `Msg` or an io error if occurred.
    async fn recv(&mut self) -> io::Result<Msg<'_>> {
        time::timeout(self.timeout, self.inner.recv())
            .await
            .map_err(|e| {
                let text = format!("Peer took too long to respond: {e}");
                io::Error::new(io::ErrorKind::TimedOut, text)
            })?
    }

    /// Sends the given messaege as is.
    ///
    /// # Args
    /// * `msg` - The message to send.
    ///
    /// # Returns
    /// An io error if occurred.
    async fn send<'a>(&mut self, msg: &Msg<'a>) -> io::Result<()> {
        self.inner.send(msg).await
    }
}
