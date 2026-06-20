use std::{io, mem};

use tokio::io::{AsyncRead, AsyncWrite};
use uuid::Uuid;

use super::{IoSwapable, TransportLayer};
use crate::{Acceptor, Connector, protocol::Msg};

/// The reconnection layer of the transport protocol.
pub struct Recon<R, W, T, F, G>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
    T: TransportLayer + IoSwapable<R, W>,
    F: Fn(R, W) -> T,
    G: AsyncFn() -> io::Result<(R, W)>,
{
    id: Uuid,
    acceptor: Acceptor<R, W, T, F, G>,
    connector: Connector<R, W, T, F>,
    inner: T,
}

impl<R, W, T, F, G> Recon<R, W, T, F, G>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
    T: TransportLayer + IoSwapable<R, W>,
    F: Fn(R, W) -> T,
    G: AsyncFn() -> io::Result<(R, W)>,
{
    /// Creates a new `Recon` transport layer.
    ///
    /// # Args
    /// * `id` - The peer's id.
    /// * `acceptor` - The acceptor to await incoming connections.
    /// * `connector` - The connector to reconnect to the peer node.
    /// * `inner` - The inner transport layer.
    ///
    /// # Returns
    /// A new `Recon` instance.
    pub fn new(
        id: Uuid,
        acceptor: Acceptor<R, W, T, F, G>,
        connector: Connector<R, W, T, F>,
        inner: T,
    ) -> Self {
        Self {
            id,
            acceptor,
            connector,
            inner,
        }
    }

    async fn reconnect(&mut self) -> io::Result<()> {
        todo!()
    }

    async fn await_reconnection(&mut self) -> io::Result<()> {
        todo!()
    }
}

impl<R, W, T, F, G> TransportLayer for Recon<R, W, T, F, G>
where
    R: AsyncRead + Unpin + Send,
    W: AsyncWrite + Unpin + Send,
    T: TransportLayer + IoSwapable<R, W>,
    F: Fn(R, W) -> T + Send,
    G: AsyncFn() -> io::Result<(R, W)> + Send,
{
    /// Tries to receive a message, if incapable it will drop the connection
    /// and will wait until receiving a reconnection from the peer.
    ///
    /// # Returns
    /// A message or an io error if occurred.
    async fn recv(&mut self) -> io::Result<Msg<'_>> {
        loop {
            match self.inner.recv().await {
                Ok(msg) => {
                    // SAFETY: The message's inner lifetime outlives '1.
                    break Ok(unsafe { mem::transmute::<Msg<'_>, Msg<'_>>(msg) });
                }
                Err(_) => {
                    self.await_reconnection().await?;
                }
            }
        }
    }

    /// Tries to send a message, if incapable will drop the connection and
    /// reconnect to the peer.
    ///
    /// # Args
    /// * `msg` - The message to send.
    ///
    /// # Returns
    /// An io error if occurred.
    async fn send<'a>(&mut self, msg: &Msg<'a>) -> io::Result<()> {
        while let Err(_) = self.inner.send(msg).await {
            self.reconnect().await?;
        }

        Ok(())
    }
}
