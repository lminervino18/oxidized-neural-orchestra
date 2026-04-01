use std::io;

use tokio::io::{AsyncWrite, AsyncWriteExt};

use super::{LEN_TYPE_SIZE, LenType, Serializer, msg::Msg};

/// The sending end handle of the communication.
pub struct OnoSender<W: AsyncWrite + Unpin> {
    tx: W,
    tx_buf: Vec<u8>,
    serializer: Serializer,
}

impl<W: AsyncWrite + Unpin> OnoSender<W> {
    /// Creates a new `OnoSender`.
    ///
    /// # Args
    /// * `tx` - The underlying writer.
    /// * `serializer` - The message serializer to use.
    ///
    /// # Returns
    /// A new `OnoSender` instance.
    pub(super) fn new(tx: W, serializer: Serializer) -> Self {
        Self {
            tx,
            tx_buf: Vec::new(),
            serializer,
        }
    }

    /// Sends `msg` through the inner sender.
    ///
    /// # Args
    /// * `msg` - The message to serialize and send.
    ///
    /// # Returns
    /// A result object that returns `io::Error` on failure.
    pub async fn send<'a>(&mut self, msg: &Msg<'a>) -> io::Result<Option<f32>> {
        let Self {
            tx,
            tx_buf,
            serializer,
        } = self;

        tx_buf.clear();
        tx_buf.resize(LEN_TYPE_SIZE, 0);

        let (zero_copy_data, threshold) = serializer.serialize(&msg, tx_buf);
        let len = tx_buf.len() - LEN_TYPE_SIZE + zero_copy_data.map(<[_]>::len).unwrap_or_default();
        let header = (len as LenType).to_be_bytes();

        tx_buf[..header.len()].copy_from_slice(&header);

        if !tx_buf.is_empty() {
            tx.write_all(tx_buf).await?;
        }

        if let Some(data) = zero_copy_data {
            tx.write_all(data).await?;
        }

        tx.flush().await?;
        Ok(threshold)
    }
}
