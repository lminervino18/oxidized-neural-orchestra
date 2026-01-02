//! The implementation of the sending end of the application layer protocol.

use std::io;

use tokio::io::{AsyncWrite, AsyncWriteExt};

use crate::{LenType, Serialize};

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
    /// Will write all it's data through `tx`.
    pub fn new(tx: W) -> Self {
        Self {
            tx,
            buf: Vec::new(),
        }
    }

    /// Sends `msg` through the inner sender.
    pub async fn send<T: Serialize>(&mut self, msg: &T) -> io::Result<()> {
        let Self { buf, tx } = self;

        buf.clear();

        let data = msg.serialize(buf).unwrap_or(buf);
        let len = data.len() as LenType;
        let header = len.to_be_bytes();

        tx.write_all(&header).await?;
        tx.write_all(data).await?;
        tx.flush().await
    }
}
