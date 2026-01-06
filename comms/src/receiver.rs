use std::io;

use tokio::io::{AsyncRead, AsyncReadExt};

use crate::{Deserialize, LEN_TYPE_SIZE, LenType};

/// The receiving end handle of the communication.
pub struct OnoReceiver<R: AsyncRead + Unpin> {
    rx: R,
    buf: Vec<u32>,
}

impl<R: AsyncRead + Unpin> OnoReceiver<R> {
    /// Creates a new `OnoReceiver` instance.
    ///
    /// # Arguments
    /// * `rx` - The underlying reader.
    pub(super) fn new(rx: R) -> Self {
        Self {
            rx,
            buf: Vec::new(),
        }
    }

    /// Waits to receive a new message from the inner receiver.
    ///
    /// # Returns
    /// A result object that returns `T` on success or `io::Error` on failure.
    pub async fn recv<'buf, T>(&'buf mut self) -> io::Result<T>
    where
        T: Deserialize<'buf>,
    {
        let Self { buf, rx } = self;

        let mut size_buf = [0; LEN_TYPE_SIZE];
        rx.read_exact(&mut size_buf).await?;
        let len = LenType::from_be_bytes(size_buf) as usize;

        let needed_u32 = len.div_ceil(4);
        if buf.len() < needed_u32 {
            buf.resize(needed_u32, 0);
        }

        let view = bytemuck::cast_slice_mut(buf);
        let slice = &mut view[..len];
        rx.read_exact(slice).await?;

        T::deserialize(slice)
    }
}
