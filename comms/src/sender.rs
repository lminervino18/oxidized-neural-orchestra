//! The implementation of the sending end of the application layer protocol.

use std::io;

use tokio::io::{AsyncWrite, AsyncWriteExt};

use crate::{LenType, Serialize};

/// The sending end handle of the communication.
pub struct OnoSender<W>
where
    W: AsyncWrite + Unpin,
{
    tx: W,
    buf: Vec<u8>,
}

impl<W: AsyncWrite + Unpin> OnoSender<W> {
    /// Creates a new `OnoSender` instance.
    ///
    /// # Arguments
    /// * `tx` - The underlying writer.
    pub(super) fn new(tx: W) -> Self {
        Self {
            tx,
            buf: Vec::new(),
        }
    }

    /// Sends `msg` through the inner sender.
    ///
    /// # Arguments
    /// * `msg` - A serializable object.
    ///
    /// # Returns
    /// A result object that returns `io::Error` on failure.
    pub async fn send<T: Serialize>(&mut self, msg: &T) -> io::Result<()> {
        let Self { buf, tx } = self;

        buf.clear();

        msg.serialize(buf);
        let len = buf.len() as LenType;
        let header = len.to_be_bytes();

        tx.write_all(&header).await?;
        tx.write_all(buf).await?;
        tx.flush().await
    }
}
