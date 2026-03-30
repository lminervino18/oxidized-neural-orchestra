use std::io;

use tokio::io::{AsyncRead, AsyncReadExt};

use crate::{Deserialize, LEN_TYPE_SIZE, LenType, msg::Msg};

const GRAD_KIND: u32 = 2;
const KIND_SIZE: usize = size_of::<u32>();

/// The receiving end handle of the communication.
pub struct OnoReceiver<R: AsyncRead + Unpin> {
    rx: R,
    buf: Vec<u32>,
}

impl<R: AsyncRead + Unpin> OnoReceiver<R> {
    /// Creates a new `OnoReceiver` instance.
    ///
    /// # Args
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
        let mut size_buf = [0; LEN_TYPE_SIZE];
        self.rx.read_exact(&mut size_buf).await?;
        let len = LenType::from_be_bytes(size_buf) as usize;

        let b_size = size_of::<u32>();
        let needed_amount = len.div_ceil(b_size);

        if self.buf.capacity() < needed_amount {
            self.buf.reserve(needed_amount - self.buf.len());
        }

        // SAFETY: The buffer has capacity for at least the amount of items. These
        //         will be immediately overwritten in the read_exact call.
        unsafe { self.buf.set_len(needed_amount) };

        let view = bytemuck::cast_slice_mut(&mut self.buf);
        let slice = &mut view[..len];
        self.rx.read_exact(slice).await?;

        T::deserialize(slice)
    }

    /// Receives the next message, dispatching on whether it is a gradient or another message type.
    ///
    /// Gradient messages are encoded as `f16` on the wire to halve network traffic.
    /// This method transparently converts the incoming `f16` values to `f32` in `grad_out`,
    /// so callers never handle the wire encoding directly.
    ///
    /// # Args
    /// * `grad_out` - Buffer to write the decoded gradient into if the next message is a gradient.
    ///               Must match the expected number of parameters exactly.
    /// * `msg_buf` - Buffer used for deserialization if the next message is not a gradient.
    ///
    /// # Returns
    /// `None` if the message was a gradient and `grad_out` was written, `Some(Msg)` otherwise.
    ///
    /// # Errors
    /// Returns an `io::Error` if the connection fails or the gradient length does not match `grad_out`.
    pub async fn recv_grad_or_msg<'buf>(
        &mut self,
        grad_out: &mut [f32],
        msg_buf: &'buf mut Vec<u32>,
    ) -> io::Result<Option<Msg<'buf>>> {
        let mut size_buf = [0u8; LEN_TYPE_SIZE];
        self.rx.read_exact(&mut size_buf).await?;
        let len = LenType::from_be_bytes(size_buf) as usize;

        let b_size = size_of::<u32>();
        let needed = len.div_ceil(b_size);

        if msg_buf.capacity() < needed {
            msg_buf.reserve(needed - msg_buf.len());
        }

        // SAFETY: The buffer has capacity for at least `needed` items.
        //         These will be immediately overwritten in the read_exact call below.
        unsafe { msg_buf.set_len(needed) };

        {
            let view: &mut [u8] = bytemuck::cast_slice_mut(msg_buf.as_mut_slice());
            self.rx.read_exact(&mut view[..len]).await?;
        }

        let kind = {
            let bytes: &[u8] = bytemuck::cast_slice(msg_buf.as_slice());
            u32::from_be_bytes(bytes[..KIND_SIZE].try_into().unwrap())
        };

        if kind == GRAD_KIND {
            let bytes: &[u8] = bytemuck::cast_slice(msg_buf.as_slice());
            let f16_bytes = &bytes[KIND_SIZE..len];
            let f16_values: &[half::f16] = bytemuck::cast_slice(f16_bytes);

            if f16_values.len() != grad_out.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "gradient size mismatch: expected {} values, got {}",
                        grad_out.len(),
                        f16_values.len()
                    ),
                ));
            }

            for (dst, &src) in grad_out.iter_mut().zip(f16_values) {
                *dst = src.to_f32();
            }

            Ok(None)
        } else {
            // SAFETY: `msg_buf` is borrowed for `'buf` and `u32` implements `Pod`,
            //         so reinterpreting as `u8` is sound. We slice to exactly `len` bytes,
            //         which is within the `needed * size_of::<u32>()` bytes allocated above.
            let byte_slice: &'buf mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(msg_buf.as_mut_ptr() as *mut u8, len)
            };
            let msg = Msg::deserialize(byte_slice)?;
            Ok(Some(msg))
        }
    }
}