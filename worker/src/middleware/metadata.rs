use comms::{OnoReceiver, OnoSender};
use tokio::io::{AsyncRead, AsyncWrite};

pub struct ServerMetadata<R, W>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    pub rx: OnoReceiver<R>,
    pub tx: OnoSender<W>,
    pub rx_buf: Vec<u32>,
    pub grad: Vec<f32>,
}
