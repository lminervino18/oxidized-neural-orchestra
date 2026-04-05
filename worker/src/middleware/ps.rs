use std::io;

use tokio::io::{AsyncRead, AsyncReadExt};

use super::{Align4, Deserialize, Deserializer, LEN_TYPE_SIZE, LenType, msg::Msg};

const GRAD_KIND: u32 = 2;
const KIND_SIZE: usize = size_of::<u32>();

/// The receiving end handle of the communication.
pub struct OnoReceiver<R: AsyncRead + Unpin> {
    rx: R,
    rx_buf: Vec<u32>,
    deserializer: Deserializer,
}

impl<R: AsyncRead + Unpin> OnoReceiver<R> {
    /// Creates a new `OnoReceiver`.
    ///
    /// # Args
    /// * `rx` - The underlying reader.
    /// * `deserializer` - The message deserializer to use.
    ///
    /// # Returns
    /// A new `OnoReceiver` instance.
    pub(super) fn new(rx: R, deserializer: Deserializer) -> Self {
        Self {
            rx,
            rx_buf: Vec::new(),
            deserializer,
        }
    }

    /// Waits to receive a new message from the inner receiver.
    ///
    /// # Returns
    /// A result object that returns `Msg` on success or `io::Error` on failure.
    pub async fn recv<'a>(&'a mut self) -> io::Result<Msg<'a>> {
        let Self {
            rx,
            rx_buf,
            deserializer,
        } = self;

        let mut size_buf = [0; LEN_TYPE_SIZE];
        rx.read_exact(&mut size_buf).await?;
        let len = LenType::from_be_bytes(size_buf) as usize;

        let b_size = size_of::<u32>();
        let needed_amount = len.div_ceil(b_size);

        if rx_buf.capacity() < needed_amount {
            rx_buf.reserve(needed_amount - rx_buf.len());
        }

        unsafe { rx_buf.set_len(needed_amount) };

        let view = bytemuck::cast_slice_mut(rx_buf);
        let slice = &mut view[..len];
        rx.read_exact(slice).await?;

        deserializer.deserialize(slice)
    }

    /// Waits to receive a new message from the inner receiver into the provided buffer.
    ///
    /// # Args
    /// * `buf` - The buffer to use for deserialization; the returned `T`'s lifetimes
    ///           will be tied to this buffer.
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

        unsafe { buf.set_len(needed_amount) };

        let view = bytemuck::cast_slice_mut(buf);
        let slice = &mut view[..len];
        self.rx.read_exact(slice).await?;

        T::deserialize(slice)
    }

    /// Receives the next message, dispatching on whether it is a gradient or another message type.
    ///
    /// This preserves the fast-path used by the f16 integration while remaining compatible
    /// with the deserializer-based receiver design.
    pub async fn recv_grad_or_msg<'buf, B: Align4>(
        &mut self,
        grad_out: &mut [f32],
        msg_buf: &'buf mut Vec<B>,
    ) -> io::Result<Option<Msg<'buf>>> {
        let mut size_buf = [0u8; LEN_TYPE_SIZE];
        self.rx.read_exact(&mut size_buf).await?;
        let len = LenType::from_be_bytes(size_buf) as usize;

        let b_size = size_of::<B>();
        let needed = len.div_ceil(b_size);

        if msg_buf.capacity() < needed {
            msg_buf.reserve(needed - msg_buf.len());
        }

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
            let byte_slice: &'buf mut [u8] =
                unsafe { std::slice::from_raw_parts_mut(msg_buf.as_mut_ptr() as *mut u8, len) };

            let msg = self.deserializer.deserialize(byte_slice)?;
            Ok(Some(msg))
        }
    }
}
