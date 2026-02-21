use comms::{OnoReceiver, OnoSender};
use tokio::io::{AsyncRead, AsyncWrite};

/// The starting size of the rx buffer.
const STARTING_RX_BUF_SIZE: usize = 1028;

/// The state of the communication with a parameter server.
pub struct ServerMetadata<R, W>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    pub rx_buf: Vec<u32>,
    pub rx: OnoReceiver<R>,
    pub tx: OnoSender<W>,
    pub grad: Vec<f32>,
    pub has_msg: bool,
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
            has_msg: false,
        }
    }
}
