use std::{io, mem, time::Duration};

use log::error;
use tokio::{
    net::tcp::{OwnedReadHalf, OwnedWriteHalf},
    time,
};
use uuid::Uuid;

use super::TransportLayer;
use crate::{Acceptor, Connector, protocol::Msg};

type R = OwnedReadHalf;
type W = OwnedWriteHalf;

/// The reconnection strategy of the `Recon` layer.
///
/// Connections coming from the `Connector`s are called `Active` connections, these will
/// try to reconnect to a peer node if the connection dies.
///
/// Connections coming from the `Acceptor` are called `Pasive` connections, these will
/// wait for reconnections from the peer node.
#[derive(Debug)]
enum Strategy {
    Active { addr: String, connector: Connector },
    Pasive { id: Uuid, acceptor: Acceptor },
}

impl Strategy {
    /// Attempts to reconnect to the peer node.
    ///
    /// # Args
    /// * `addr` - The peer's network address.
    /// * `connector` - A connector to attempt reconnections.
    /// * `layer` - The inner transport layer.
    ///
    /// # Returns
    /// An io error if occurred.
    async fn active_reconnect<T>(addr: &str, connector: &Connector, layer: &mut T) -> io::Result<()>
    where
        T: TransportLayer<R, W>,
    {
        const BASE_DELAY: Duration = Duration::from_secs(5);
        const RETRY_COEF: u32 = 2;
        const MAX_EXP: u32 = 5;

        let mut sleep_dur = Duration::from_secs(5);
        let mut exp = 0;

        while let Err(e) = connector.reconnect(addr, layer).await {
            error!("Failed to reconnect to {addr}: {e}");
            time::sleep(sleep_dur).await;
            sleep_dur = BASE_DELAY * RETRY_COEF.pow(exp.min(MAX_EXP));
            exp += 1;
        }

        Ok(())
    }

    /// Waits until the other side connects to this node using the inner acceptor.
    ///
    /// # Args
    /// * `id` - The peer's id.
    /// * `acceptor` - The acceptor to await incoming connections.
    /// * `layer` - The inner transport layer.
    ///
    /// # Returns
    /// An io error if occurred.
    async fn passive_reconnect<T>(id: Uuid, acceptor: &Acceptor, layer: &mut T) -> io::Result<()>
    where
        T: TransportLayer<R, W>,
    {
        acceptor.reconnect(id, layer).await
    }

    /// Dispatches to the correct reconnection strategy.
    ///
    /// # Args
    /// * `layer` - The inner transport layer.
    ///
    /// # Returns
    /// An io error if occurred.
    async fn reconnect<T>(&self, layer: &mut T) -> io::Result<()>
    where
        T: TransportLayer<R, W>,
    {
        match self {
            Strategy::Active { addr, connector } => {
                Self::active_reconnect(addr, connector, layer).await
            }
            Strategy::Pasive { id, acceptor } => {
                Self::passive_reconnect(*id, acceptor, layer).await
            }
        }
    }
}

/// The reconnection layer of the transport protocol.
#[derive(Debug)]
pub struct Recon<T>
where
    T: TransportLayer<R, W>,
{
    strat: Strategy,
    inner: T,
}

impl<T> Recon<T>
where
    T: TransportLayer<R, W>,
{
    /// Creates a new `Recon` transport layer in it's `Active` role.
    ///
    /// # Args
    /// * `addr` - The peer's network address.
    /// * `connector` - A connector to attempt reconnections.
    /// * `inner` - The inner transport layer.
    ///
    /// # Returns
    /// A new `Recon` instance.
    pub fn active(addr: String, connector: Connector, inner: T) -> Self {
        Self {
            strat: Strategy::Active { addr, connector },
            inner,
        }
    }

    /// Creates a new `Recon` transport layer in it's `Pasive` role.
    ///
    /// # Args
    /// * `id` - The peer's id.
    /// * `acceptor` - The acceptor to await incoming connections.
    /// * `inner` - The inner transport layer.
    ///
    /// # Returns
    /// A new `Recon` instance.
    pub fn passive(id: Uuid, acceptor: Acceptor, inner: T) -> Self {
        Self {
            strat: Strategy::Pasive { id, acceptor },
            inner,
        }
    }
}

impl<T> TransportLayer<R, W> for Recon<T>
where
    T: TransportLayer<R, W> + Sync + Send,
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
                    let _ = self.strat.reconnect(&mut self.inner).await;
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
            let _ = self.strat.reconnect(&mut self.inner).await;
        }

        Ok(())
    }

    fn swap(&mut self, reader: R, writer: W) {
        self.inner.swap(reader, writer);
    }

    fn demount(self) -> (R, W) {
        self.inner.demount()
    }
}
