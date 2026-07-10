mod framer;
mod layer;

pub use framer::Framer;
pub use layer::TransportLayer;
use tokio::{
    io::{AsyncRead, AsyncWrite},
    net::tcp::{OwnedReadHalf, OwnedWriteHalf},
};

/// The simple transport;
pub type Stp<R, W> = Framer<R, W>;

/// The network TCP reliable transport layer.
pub type NetStp = Stp<OwnedReadHalf, OwnedWriteHalf>;

/// Builds an uninitialized simple transport.
///
/// # Args
/// * `reader` - The reading end of the communication.
/// * `writer` - The writing end of the communication.
///
/// # Returns
/// An uninitialized `Stp` instance.
pub fn build_simple_transport<R, W>(reader: R, writer: W) -> Stp<R, W>
where
    R: AsyncRead + Unpin + Send,
    W: AsyncWrite + Unpin + Send,
{
    let inner = Framer::new(reader, writer);
    inner
}
