use comms::{OnoReceiver, OnoSender};
use tokio::io::{AsyncRead, AsyncWrite};

/// The starting size of the rx buffer.
const STARTING_RX_BUF_SIZE: usize = 1028;

/// The server's metadata to maintain while the training is going on.
pub struct ServerMetadata<R, W>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    pub rx_buf: Vec<u32>,
    pub rx: OnoReceiver<R>,
    pub tx: OnoSender<W>,
    pub grad: Vec<f32>,
}

impl<R, W> ServerMetadata<R, W>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    /// Creates a new `ServerMetadata`.
    ///
    /// # Arguments
    /// * `rx` - The worker's server receiver.
    /// * `tx` - The worker's server sender.
    /// * `size` - The size of the server's storage.
    ///
    /// # Returns
    /// A new `ServerMetadata` instance.
    pub fn new(rx: OnoReceiver<R>, tx: OnoSender<W>, size: usize) -> Self {
        Self {
            rx_buf: vec![0; STARTING_RX_BUF_SIZE],
            rx,
            tx,
            grad: vec![0.0; size],
        }
    }
}

/// The state necessary to make forward and backward passes through the network.
pub struct ServerParamsMetadata<'a> {
    pub params: &'a mut [f32],
    pub grad: &'a mut [f32],
}
