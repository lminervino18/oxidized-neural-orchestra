use std::io;

use tokio::io::{AsyncRead, AsyncReadExt};

use crate::{Deserialize, LEN_TYPE_SIZE, LenType};

/// The receiving end handle of the communication.
pub struct OnoReceiver<R: AsyncRead + Unpin> {
    rx: R,
    rx_buf: Vec<u32>,
}

impl<R: AsyncRead + Unpin> OnoReceiver<R> {
    /// Creates a new `OnoReceiver` instance.
    ///
    /// # Args
    /// * `rx` - The underlying reader.
    pub(super) fn new(rx: R) -> Self {
        Self {
            rx,
            rx_buf: Vec::new(),
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
        let Self { rx, rx_buf, .. } = self;

        let mut size_buf = [0; LEN_TYPE_SIZE];
        rx.read_exact(&mut size_buf).await?;
        let len = LenType::from_be_bytes(size_buf) as usize;

        let b_size = size_of::<u32>();
        let needed_amount = len.div_ceil(b_size);

        if rx_buf.capacity() < needed_amount {
            rx_buf.reserve(needed_amount - rx_buf.len());
        }

        // SAFETY: The buffer has capacity for at least the amount of items. These
        //         will be immediatelly overwritten in the read_exact call.
        unsafe { rx_buf.set_len(needed_amount) };

        let view = bytemuck::cast_slice_mut(rx_buf);
        let slice = &mut view[..len];
        rx.read_exact(slice).await?;

        T::deserialize(slice)
    }
}
