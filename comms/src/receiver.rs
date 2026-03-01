use std::io;

use tokio::io::{AsyncRead, AsyncReadExt};

use crate::{Align4, Deserialize, LEN_TYPE_SIZE, LenType};

/// The receiving end handle of the communication.
pub struct OnoReceiver<R: AsyncRead + Unpin> {
    rx: R,
}

impl<R: AsyncRead + Unpin> OnoReceiver<R> {
    /// Creates a new `OnoReceiver` instance.
    ///
    /// # Arguments
    /// * `rx` - The underlying reader.
    pub(super) fn new(rx: R) -> Self {
        Self { rx }
    }

    /// Waits to receive a new message from the inner receiver.
    ///
    /// # Arguments
    /// * `buf` - The buffer to use for deserialization, the returned
    ///           `T`'s lifetimes will be tied to this buffer.
    ///
    /// # Returns
    /// A result object that returns `T` on success or `io::Error` on failure.
    pub async fn recv_into<'buf, T, B>(&mut self, buf: &'buf mut Vec<B>) -> io::Result<T>
    where
        T: Deserialize<'buf>,
        B: Align4,
    {
        let mut size_buf = [0; LEN_TYPE_SIZE];
        self.rx.read_exact(&mut size_buf).await?;
        let len = LenType::from_be_bytes(size_buf) as usize;

        let b_size = size_of::<B>();
        let needed_amount = len.div_ceil(b_size);

        if buf.capacity() < needed_amount {
            buf.reserve(needed_amount - buf.len());
        }

        // SAFETY: The buffer has capacity for at least the amount of items. These
        //         will be immediatelly overwritten in the read_exact call.
        unsafe { buf.set_len(needed_amount) };

        let view = bytemuck::cast_slice_mut(buf);
        let slice = &mut view[..len];
        self.rx.read_exact(slice).await?;

        T::deserialize(slice)
    }
}
