use std::io;

use comms::{
    OnoReceiver, OnoSender,
    msg::{Command, Msg, Payload},
};
use tokio::io::{AsyncRead, AsyncWrite};

const STARTING_RX_BUF_SIZE: usize = 1028;

struct PeerMetadata<R, W>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    worker_id: usize,
    rx: OnoReceiver<R>,
    tx: OnoSender<W>,
    rx_buf: Vec<u32>,
}

/// The communication manager between a worker and its ring neighbors.
pub struct RingMiddleware<R, W>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    prev: PeerMetadata<R, W>,
    next: PeerMetadata<R, W>,
}

impl<R, W> RingMiddleware<R, W>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    /// Creates a new `RingMiddleware`.
    ///
    /// # Args
    /// * `prev_worker_id` - The worker id of the previous ring neighbor.
    /// * `prev_rx` - The receiving end of the connection from the previous ring neighbor.
    /// * `prev_tx` - The sending end of the connection from the previous ring neighbor.
    /// * `next_worker_id` - The worker id of the next ring neighbor.
    /// * `next_rx` - The receiving end of the connection to the next ring neighbor.
    /// * `next_tx` - The sending end of the connection to the next ring neighbor.
    ///
    /// # Returns
    /// A new `RingMiddleware` instance.
    pub fn new(
        prev_worker_id: usize,
        prev_rx: OnoReceiver<R>,
        prev_tx: OnoSender<W>,
        next_worker_id: usize,
        next_rx: OnoReceiver<R>,
        next_tx: OnoSender<W>,
    ) -> Self {
        Self {
            prev: PeerMetadata {
                worker_id: prev_worker_id,
                rx: prev_rx,
                tx: prev_tx,
                rx_buf: vec![0; STARTING_RX_BUF_SIZE],
            },
            next: PeerMetadata {
                worker_id: next_worker_id,
                rx: next_rx,
                tx: next_tx,
                rx_buf: vec![0; STARTING_RX_BUF_SIZE],
            },
        }
    }

    /// Sends a chunk to the next ring neighbor.
    ///
    /// # Args
    /// * `chunk` - The chunk to send to the next ring neighbor.
    ///
    /// # Errors
    /// Returns an io error if the chunk cannot be sent.
    pub async fn send_next_chunk(&mut self, chunk: &[f32]) -> io::Result<()> {
        let msg = Msg::Data(Payload::Datachunk(chunk));
        self.next.tx.send(&msg).await
    }

    /// Receives a chunk from the previous ring neighbor.
    ///
    /// # Returns
    /// The received chunk.
    ///
    /// # Errors
    /// Returns an io error if the chunk cannot be received or decoded.
    pub async fn recv_prev_chunk(&mut self) -> io::Result<&[f32]> {
        match self.prev.rx.recv_into(&mut self.prev.rx_buf).await? {
            Msg::Data(Payload::Datachunk(chunk)) => Ok(chunk),
            msg => Err(io::Error::other(format!(
                "expected ring chunk from worker {}, got: {msg:?}",
                self.prev.worker_id
            ))),
        }
    }

    /// Disconnects this worker from both ring neighbors.
    ///
    /// # Errors
    /// Returns an io error if any disconnect message fails.
    pub async fn disconnect(&mut self) -> io::Result<()> {
        let msg = Msg::Control(Command::Disconnect);
        self.prev.tx.send(&msg).await?;
        self.next.tx.send(&msg).await?;
        Ok(())
    }
}
