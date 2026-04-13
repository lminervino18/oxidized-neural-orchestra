use std::io;

use tokio::io::{AsyncWrite, AsyncWriteExt};

use super::{LEN_TYPE_SIZE, LenType};
use crate::protocol::Msg;

/// The sending end handle of the communication.
pub struct Sink<W: AsyncWrite + Unpin> {
    writer: W,
    buf: Vec<u8>,
}

impl<W: AsyncWrite + Unpin> Sink<W> {
    /// Creates a new `Sink`.
    ///
    /// # Args
    /// * `writer` - The underlying writer.
    ///
    /// # Returns
    /// A new `Sink` instance.
    pub fn new(writer: W) -> Self {
        Self {
            writer,
            buf: Vec::new(),
        }
    }

    /// Writes the msg prefixed by the payload's length.
    ///
    /// # Args
    /// * `msg` - The message to serialize and send.
    ///
    /// # Returns
    /// An io error if occurred.
    pub async fn send<'a>(&mut self, msg: &Msg<'a>) -> io::Result<()> {
        let Self { writer, buf } = self;

        buf.clear();
        buf.resize(LEN_TYPE_SIZE, 0);

        let zero_copy_data = msg.serialize(buf);
        let len = buf.len() - LEN_TYPE_SIZE + zero_copy_data.map(<[_]>::len).unwrap_or_default();
        let header = (len as LenType).to_be_bytes();

        buf[..header.len()].copy_from_slice(&header);

        if !buf.is_empty() {
            writer.write_all(buf).await?;
        }

        if let Some(data) = zero_copy_data {
            writer.write_all(data).await?;
        }

        writer.flush().await
    }
}
