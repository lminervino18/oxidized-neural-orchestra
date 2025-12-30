//! The implementation of the sending end of the application layer protocol.

use std::io::{self, IoSlice};

use bytes::BytesMut;
use tokio::io::{AsyncWrite, AsyncWriteExt};

use crate::Serialize;

pub struct OnoSender<W: AsyncWrite + Unpin> {
    tx: W,
    buf: BytesMut,
}

impl<W: AsyncWrite + Unpin> OnoSender<W> {
    /// Creates a new `OnoSender` instance.
    ///
    /// # Arguments
    /// * `tx` - The underlying writer.
    pub fn new(tx: W) -> Self {
        Self {
            tx,
            buf: BytesMut::with_capacity(64 * 1024),
        }
    }

    /// Sends `msg` through the inner sender.
    ///
    /// # Arguments
    /// `msg` - The message to be serialized and sent.
    pub async fn send<T: Serialize>(&mut self, msg: &T) -> io::Result<usize> {
        self.buf.clear();
        msg.serialize(&mut self.buf)?;

        let len = self.buf.len() as u64;
        let header = len.to_be_bytes();

        let slices = [IoSlice::new(&header), IoSlice::new(&self.buf)];
        self.tx.write_vectored(&slices).await
    }
}
