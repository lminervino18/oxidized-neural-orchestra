use comms::{OnoReceiver, OnoSender};
use tokio::io::{AsyncRead, AsyncWrite};

/// The starting size of the receiver buffer.
const STARTING_RX_BUF_SIZE: usize = 1028;

/// The necessary information to maintain for the entire
/// training duration for each of the servers.
pub struct ServerMetadata<R, W>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    pub rx: OnoReceiver<R>,
    pub tx: OnoSender<W>,
    pub rx_buf: Vec<u32>,
    pub grad: Vec<f32>,
    pub acc_grad_buf: Vec<f32>,
}

impl<R, W> ServerMetadata<R, W>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    /// Creates a new `ServerMetadata`.
    ///
    /// # Arguments
    /// * `rx` - The receiving end of the communication.
    /// * `tx` - The sending end of the communication.
    /// * `size` - The amount of parameters this server holds.
    ///
    /// # Returns
    /// A new `ServerMetadata` instance.
    pub fn new(rx: OnoReceiver<R>, tx: OnoSender<W>, size: usize) -> Self {
        Self {
            rx,
            tx,
            rx_buf: vec![0; STARTING_RX_BUF_SIZE],
            grad: vec![0.0; size],
            acc_grad_buf: vec![0.0; size],
        }
    }
}
