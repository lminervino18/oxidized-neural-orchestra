use std::{io, time::Duration};

use super::TransportLayer;
use crate::protocol::Msg;

/// The `TimeOuter` tries receiving messages inside a time window.
/// If it fails it returns an error with `ErrorKind::TimedOut`.
///
/// Note: the receive timeout is currently disabled, so this layer is a
/// pass-through over its inner transport. The wrapper is kept in the stack so
/// the timeout can be re-enabled without reshaping the transport types.
#[derive(Debug)]
pub struct TimeOuter<L: TransportLayer> {
    inner: L,
}

impl<L: TransportLayer> TimeOuter<L> {
    /// Creates a new `TimeOuter` transport layer.
    ///
    /// # Args
    /// * `timeout` - Receive timeout (currently unused; the layer is a pass-through).
    /// * `inner` - The inner transport layer stack.
    ///
    /// # Returns
    /// A new `TimeOuter` transport layer instance.
    pub fn new(_timeout: Duration, inner: L) -> Self {
        Self { inner }
    }
}

impl<L: TransportLayer> TransportLayer for TimeOuter<L> {
    /// Calls receive on the inner transport layer.
    ///
    /// # Returns
    /// A deserialized `Msg` or an io error if occurred.
    async fn recv(&mut self) -> io::Result<Msg<'_>> {
        // Pass-through: the receive timeout is currently disabled.
        self.inner.recv().await
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
