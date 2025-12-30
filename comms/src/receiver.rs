//! The implementation of the receiving end of the application layer protocol.

use std::io;

use bytes::{Buf, BytesMut};
use tokio::io::{AsyncRead, AsyncReadExt};

use crate::Deserialize;

pub struct OnoReceiver<R: AsyncRead + Unpin> {
    rx: R,
    buf: BytesMut,
    max_frame_size: usize,
}

impl<R: AsyncRead + Unpin> OnoReceiver<R> {
    /// Creates a new `OnoReceiver` instance.
    ///
    /// # Arguments
    /// * `rx` - The underlying reader.
    /// * `max_frame_size` - The maximum size of the communication frame.
    pub fn new(rx: R, max_frame_size: usize) -> Self {
        Self {
            rx,
            buf: BytesMut::with_capacity(64 * 1024),
            max_frame_size,
        }
    }

    /// Waits to receive a new message from the inner receiver.
    pub async fn recv<T: Deserialize>(&mut self) -> io::Result<T> {
        let mut header = [0u8; 8];
        self.rx.read_exact(&mut header).await?;
        let len = u64::from_be_bytes(header) as usize;

        if len > self.max_frame_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Frame size {} exceeds limit {}", len, self.max_frame_size),
            ));
        }

        self.buf.clear();
        self.buf.reserve(len);

        let mut limited = (&mut self.rx).take(len as u64);

        while self.buf.len() < len {
            if limited.read_buf(&mut self.buf).await? == 0 {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "Early EOF"));
            }
        }

        let msg = T::deserialize(&mut self.buf)?;

        if self.buf.has_remaining() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Trailing bytes"));
        }

        Ok(msg)
    }
}
