use std::io;

use tokio::io::{AsyncRead, AsyncReadExt};

use super::{LEN_TYPE_SIZE, LenType};
use crate::protocol::Msg;

/// The receiving end handle of the communication.
pub struct Source<R: AsyncRead + Unpin> {
    reader: R,
    buf: Vec<u32>,
}

impl<R: AsyncRead + Unpin> Source<R> {
    /// Creates a new `Source`.
    ///
    /// # Args
    /// * `reader` - The underlying reader.
    ///
    /// # Returns
    /// A new `Source` instance.
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            buf: Vec::new(),
        }
    }

    /// Waits to receive a new message from the inner reader.
    ///
    /// # Returns
    /// A result object that returns `T` on success or `io::Error` on failure.
    pub async fn recv<'a>(&'a mut self) -> io::Result<Msg<'a>> {
        let Self { reader, buf } = self;

        let mut size_buf = [0; LEN_TYPE_SIZE];
        reader.read_exact(&mut size_buf).await?;
        let len = LenType::from_be_bytes(size_buf) as usize;

        let b_size = size_of::<u32>();
        let needed_amount = len.div_ceil(b_size);

        if buf.capacity() < needed_amount {
            buf.reserve(needed_amount - buf.len());
        }

        // SAFETY: The buffer has capacity for at least the amount of items. These
        //         will be immediatelly overwritten in the read_exact call.
        unsafe { buf.set_len(needed_amount) };

        let view = bytemuck::cast_slice_mut(buf);
        let slice = &mut view[..len];
        reader.read_exact(slice).await?;

        Msg::deserialize(slice)
    }
}
