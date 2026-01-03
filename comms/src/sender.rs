//! The implementation of the sending end of the application layer protocol.

use std::io;

use tokio::io::{AsyncWrite, AsyncWriteExt};

use crate::{LEN_TYPE_SIZE, LenType, Serialize};

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
    pub async fn send<'a, T: Serialize<'a>>(&mut self, msg: &'a T) -> io::Result<()> {
        let Self { buf, tx } = self;

        buf.clear();
        buf.resize(LEN_TYPE_SIZE, 0);

        let zero_copy_data = msg.serialize(buf);
        let len = buf.len() - LEN_TYPE_SIZE + zero_copy_data.map(<[_]>::len).unwrap_or_default();
        let header = (len as LenType).to_be_bytes();

        buf[..header.len()].copy_from_slice(&header);

        if !buf.is_empty() {
            tx.write_all(buf).await?;
        }

        if let Some(data) = zero_copy_data {
            tx.write_all(data).await?;
        }

        tx.flush().await
    }
}
