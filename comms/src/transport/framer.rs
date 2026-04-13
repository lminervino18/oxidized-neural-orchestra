use std::io;

use tokio::io::{AsyncRead, AsyncWrite};

use super::TransportLayer;
use crate::{
    codec::{Sink, Source},
    protocol::Msg,
};

/// The `Framer` builds framed messages by prefixing the payload with it's size.
///
/// It's the last layer of the entire transport layer stack.
pub struct Framer<R, W>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    rx: Source<R>,
    tx: Sink<W>,
}

impl<R, W> Framer<R, W>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    /// Creates a new `Framer` transport layer.
    ///
    /// # Args
    /// * `reader` - The underlying readable.
    /// * `writer` - The underlying writable.
    ///
    /// # Returns
    /// A new `Framer` transport layer instance.
    pub fn new(reader: R, writer: W) -> Self {
        Self {
            rx: Source::new(reader),
            tx: Sink::new(writer),
        }
    }
}

impl<R, W> TransportLayer for Framer<R, W>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    /// Waits to receive a new message from the inner receiver.
    ///
    /// # Returns
    /// A deserialized `Msg` or an io error if occurred.
    async fn recv(&mut self) -> io::Result<Msg<'_>> {
        self.rx.recv().await
    }

    /// Sends `msg` through the inner sender.
    ///
    /// # Args
    /// * `msg` - The message to serialize and send.
    ///
    /// # Returns
    /// An io error if occurred.
    async fn send<'a>(&mut self, msg: &Msg<'a>) -> io::Result<()> {
        self.tx.send(msg).await
    }
}
