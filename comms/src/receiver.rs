use std::io;

use tokio::io::{AsyncRead, AsyncReadExt};

use super::{Deserializer, LEN_TYPE_SIZE, LenType, msg::Msg};

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
    /// A result object that returns `T` on success or `io::Error` on failure.
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

        // SAFETY: The buffer has capacity for at least the amount of items. These
        //         will be immediatelly overwritten in the read_exact call.
        unsafe { rx_buf.set_len(needed_amount) };

        let view = bytemuck::cast_slice_mut(rx_buf);
        let slice = &mut view[..len];
        rx.read_exact(slice).await?;

        deserializer.deserialize(slice)
    }

    /// Waits to receive a new message from the inner receiver.
    ///
    /// # Args
    /// * `buf` - The buffer to use for deserialization; the returned `T`'s lifetimes
    ///           will be tied to this buffer.
    ///
    /// # Returns
    /// A result object that returns `T` on success or `io::Error` on failure.
    pub async fn recv_into<'a>(&'a mut self, buf: &'a mut Vec<u32>) -> io::Result<Msg<'a>> {
        let mut size_buf = [0; LEN_TYPE_SIZE];
        self.rx.read_exact(&mut size_buf).await?;
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
        self.rx.read_exact(slice).await?;

        self.deserializer.deserialize(slice)
    }

    /// Receives the next message, dispatching on whether it is a gradient or another message type.
    ///
    /// This preserves the fast-path used by the f16 integration while remaining compatible
    /// with the deserializer-based receiver design.
    pub async fn recv_grad_or_msg<'a>(
        &'a mut self,
        grad_out: &mut [f32],
        msg_buf: &'a mut Vec<u32>,
    ) -> io::Result<Option<Msg<'a>>> {
        match self.recv_into(msg_buf).await? {
            Msg::Data(super::msg::Payload::Grad(grad)) => {
                if grad_out.len() != grad.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "gradient size mismatch: expected {} values, got {}",
                            grad_out.len(),
                            grad.len()
                        ),
                    ));
                }

                grad_out.copy_from_slice(grad);
                Ok(None)
            }
            msg => Ok(Some(msg)),
        }
    }
}
