use std::{io, marker::PhantomData, time::Duration};

use tokio::{
    io::{AsyncRead, AsyncWrite},
    time,
};

use super::TransportLayer;
use crate::protocol::Msg;

/// The `TimeOuter` tries receiving messages inside a time window.
/// If it fails it returns an error with `ErrorKind::TimedOut`.
#[derive(Debug)]
pub struct TimeOuter<R, W, T>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
    T: TransportLayer<R, W>,
{
    timeout: Duration,
    inner: T,
    _phantom: PhantomData<(R, W)>,
}

impl<R, W, T> TimeOuter<R, W, T>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
    T: TransportLayer<R, W>,
{
    /// Creates a new `TimeOuter` transport layer.
    ///
    /// # Args
    /// * `timeout` - The duration to wait until receiving a message and declaring an error.
    /// * `inner` - The inner transport layer stack.
    ///
    /// # Returns
    /// A new `TimeOuter` transport layer instance.
    pub fn new(timeout: Duration, inner: T) -> Self {
        Self {
            timeout,
            inner,
            _phantom: PhantomData,
        }
    }
}

impl<R, W, T> TransportLayer<R, W> for TimeOuter<R, W, T>
where
    R: AsyncRead + Unpin + Send,
    W: AsyncWrite + Unpin + Send,
    T: TransportLayer<R, W>,
{
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

    fn swap(&mut self, reader: R, writer: W) {
        self.inner.swap(reader, writer);
    }

    fn demount(self) -> (R, W) {
        self.inner.demount()
    }
}
