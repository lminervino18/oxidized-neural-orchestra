use std::{io, mem, time::Duration};

use log::error;
use tokio::{
    net::{
        TcpStream,
        tcp::{OwnedReadHalf, OwnedWriteHalf},
    },
    time,
};
use uuid::Uuid;

use super::{Demountable, IoSwapable, TransportLayer};
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
enum Strategy<T, F, G, Fut>
where
    T: TransportLayer + IoSwapable<R, W>,
    F: Fn(R, W) -> T,
    G: Fn() -> Fut,
    Fut: Future<Output = io::Result<(R, W)>>,
{
    Active {
        addr: String,
        connector: Connector<R, W, T, F>,
    },
    Pasive {
        id: Uuid,
        acceptor: Acceptor<R, W, T, F, G, Fut>,
    },
}

impl<T, F, G, Fut> Strategy<T, F, G, Fut>
where
    T: TransportLayer + IoSwapable<R, W> + Demountable<R, W>,
    F: Fn(R, W) -> T,
    G: Fn() -> Fut,
    Fut: Future<Output = io::Result<(R, W)>>,
{
    /// Attempts to reconnect to the peer node.
    ///
    /// # Args
    /// * `addr` - The peer's network address.
    /// * `connector` - A connector to attempt reconnections.
    /// * `layer` - The inner transport layer.
    ///
    /// # Returns
    /// An io error if occurred.
    async fn active_reconnect(
        addr: &str,
        connector: &Connector<R, W, T, F>,
        layer: &mut T,
    ) -> io::Result<()> {
        const BASE_DELAY: Duration = Duration::from_secs(5);
        const RETRY_COEF: u32 = 2;
        const MAX_EXP: u32 = 5;

        let mut connect = async || {
            let stream = TcpStream::connect(addr).await?;
            let (rx, tx) = stream.into_split();
            connector.reconnect(rx, tx, layer).await
        };

        let mut sleep_dur = Duration::from_secs(5);
        let mut exp = 0;

        while let Err(e) = connect().await {
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
    async fn passive_reconnect(
        id: Uuid,
        acceptor: &Acceptor<R, W, T, F, G, Fut>,
        layer: &mut T,
    ) -> io::Result<()> {
        acceptor.reconnect(id, layer).await
    }

    /// Dispatches to the correct reconnection strategy.
    ///
    /// # Args
    /// * `layer` - The inner transport layer.
    ///
    /// # Returns
    /// An io error if occurred.
    async fn reconnect(&self, layer: &mut T) -> io::Result<()> {
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
pub struct Recon<T, F, G, Fut>
where
    T: TransportLayer + IoSwapable<R, W>,
    F: Fn(R, W) -> T,
    G: Fn() -> Fut,
    Fut: Future<Output = io::Result<(R, W)>>,
{
    strat: Strategy<T, F, G, Fut>,
    inner: T,
}

impl<T, F, G, Fut> Recon<T, F, G, Fut>
where
    T: TransportLayer + IoSwapable<R, W> + Demountable<R, W>,
    F: Fn(R, W) -> T,
    G: Fn() -> Fut,
    Fut: Future<Output = io::Result<(R, W)>>,
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
    pub fn active(addr: String, connector: Connector<R, W, T, F>, inner: T) -> Self {
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
    pub fn passive(id: Uuid, acceptor: Acceptor<R, W, T, F, G, Fut>, inner: T) -> Self {
        Self {
            strat: Strategy::Pasive { id, acceptor },
            inner,
        }
    }
}

impl<T, F, G, Fut> TransportLayer for Recon<T, F, G, Fut>
where
    T: TransportLayer + IoSwapable<R, W> + Demountable<R, W> + Sync,
    F: Fn(R, W) -> T + Send + Sync,
    G: Fn() -> Fut + Send + Sync,
    Fut: Future<Output = io::Result<(R, W)>> + Send,
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
}
