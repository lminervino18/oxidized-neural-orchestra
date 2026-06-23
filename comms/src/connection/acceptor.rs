use std::{collections::HashMap, io, marker::PhantomData, sync::Arc};

use tokio::{
    net::tcp::{OwnedReadHalf, OwnedWriteHalf},
    sync::{Mutex, Notify},
};
use uuid::Uuid;

use super::Connection;
use crate::{
    NodeHandle, OrchHandle, ParamServerHandle, WorkerHandle,
    protocol::{Command, Entity, Msg},
    transport::TransportLayer,
};

type R = OwnedReadHalf;
type W = OwnedWriteHalf;

/// The shared state for managing incoming connections.
#[derive(Default)]
struct ConnRegistry {
    backlog: Mutex<HashMap<Uuid, (R, W)>>,
    notify: Notify,
}

/// Accepts new incoming connections and assigns yields their handle types.
pub struct Acceptor<T, F, G, H, Fut>
where
    T: TransportLayer<R, W>,
    F: Fn(R, W) -> T + Clone,
    G: Fn() -> Fut + Clone,
    H: Fn(Uuid, Self, T) -> T,
    Fut: Future<Output = io::Result<(R, W)>>,
{
    id: Uuid,
    conns: Arc<ConnRegistry>,
    transport_factory: F,
    connection_factory: G,
    passive_recon_wrapper: Arc<H>,
    _phantom: PhantomData<(T, G)>,
}

impl<T, F, G, H, Fut> Clone for Acceptor<T, F, G, H, Fut>
where
    T: TransportLayer<R, W>,
    F: Fn(R, W) -> T + Clone,
    G: Fn() -> Fut + Clone,
    H: Fn(Uuid, Self, T) -> T,
    Fut: Future<Output = io::Result<(R, W)>>,
{
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            conns: Arc::clone(&self.conns),
            transport_factory: self.transport_factory.clone(),
            connection_factory: self.connection_factory.clone(),
            passive_recon_wrapper: Arc::clone(&self.passive_recon_wrapper),
            _phantom: self._phantom,
        }
    }
}

impl<T, F, G, H, Fut> Acceptor<T, F, G, H, Fut>
where
    T: TransportLayer<R, W>,
    F: Fn(R, W) -> T + Clone,
    G: Fn() -> Fut + Clone,
    H: Fn(Uuid, Self, T) -> T,
    Fut: Future<Output = io::Result<(R, W)>>,
{
    /// Creates a new `Acceptor`.
    ///
    /// # Args
    /// * `id` - This node's id.
    /// * `connection_factory` - A closure that waits for new connections.
    /// * `transport_factory` - A transport layer factory closure.
    /// * `recon_wrapper` - Takes a reliable transport and wraps it with a passive reconnection layer.
    ///
    /// # Returns
    /// A new `Acceptor` instance.
    pub fn new(id: Uuid, transport_factory: F, connection_factory: G, recon_wrapper: H) -> Self {
        Self {
            id,
            conns: Default::default(),
            transport_factory,
            connection_factory,
            passive_recon_wrapper: Arc::new(recon_wrapper),
            _phantom: PhantomData,
        }
    }

    /// Blocks the current thread until a new connection arrives.
    ///
    /// # Args
    /// * `src` - This node's entity variant.
    ///
    /// # Returns
    /// A new connection or an io error if occurred while waiting for incoming connections
    /// or receiving the type of entity from the peer.
    pub async fn accept(&self, src: Entity) -> io::Result<Connection<R, W, T>> {
        let (rx, tx) = (self.connection_factory)().await?;
        let mut layer = (self.transport_factory)(rx, tx);
        let (id, dst) = self.handshake(src, &mut layer).await?;

        let conn = match dst {
            Entity::Worker => Connection::Worker(WorkerHandle::new(id, layer)),
            Entity::ParamServer => Connection::ParamServer(ParamServerHandle::new(id, layer)),
            Entity::Orchestrator => Connection::Orch(OrchHandle::new(id, layer)),
            Entity::Node => Connection::Node(NodeHandle::new(id, layer)),
            entity => {
                let details = format!("Accepted invalid incoming connection from {entity:?}");
                return Err(io::Error::other(details));
            }
        };

        Ok(conn)
    }

    /// Makes the handshake with the peer sharing the entity types and ids.
    ///
    /// # Args
    /// * `src` - This node's entity variant.
    /// * `layer` - The transport layer used to communicate with the other end.
    ///
    /// # Returns
    /// The peer's id and entity type or an io error if occurred.
    async fn handshake(&self, src: Entity, layer: &mut T) -> io::Result<(Uuid, Entity)> {
        let msg = layer.recv().await?;

        let Msg::Control(Command::Connect { id, src: dst }) = msg else {
            let text = format!("Expected Connect message, got: {msg:?}");
            return Err(io::Error::other(text));
        };

        let msg = Msg::Control(Command::Accept { id: self.id, src });
        layer.send(&msg).await?;

        Ok((id, dst))
    }
}

impl<T, F, G, H, Fut> Acceptor<T, F, G, H, Fut>
where
    T: TransportLayer<R, W>,
    F: Fn(R, W) -> T + Clone,
    G: Fn() -> Fut + Clone,
    H: Fn(Uuid, Self, T) -> T,
    Fut: Future<Output = io::Result<(R, W)>>,
{
    /// Will wait until a connection with the expected `id` arrives blocking the thread in the process.
    ///
    /// # Args
    /// * `id` - The id that's being waited for.
    /// * `layer` - The transport layer used to communicate with the other end.
    ///
    /// # Returns
    /// An io error if occurred.
    pub async fn reconnect(&self, id: Uuid, layer: &mut T) -> io::Result<()> {
        const RECONNECT: Entity = Entity::Reconnect;

        let (rx, tx) = loop {
            {
                let mut backlog = self.conns.backlog.lock().await;

                if let Some(rx_tx) = backlog.remove(&id) {
                    break rx_tx;
                }
            }

            let (rx, tx) = tokio::select! {
                _ = self.conns.notify.notified() => continue,
                res = (self.connection_factory)() => res?,
            };

            let mut tmp = (self.transport_factory)(rx, tx);
            let (peer_id, dst) = self.handshake(RECONNECT, &mut tmp).await?;

            if dst != RECONNECT {
                let details = format!("Expected {RECONNECT:?}, got: {dst:?} with id {peer_id}");
                return Err(io::Error::other(details));
            }

            let rx_tx = tmp.demount();

            if peer_id == id {
                break rx_tx;
            }

            let mut backlog = self.conns.backlog.lock().await;
            backlog.insert(peer_id, rx_tx);
            self.conns.notify.notify_waiters();
        };

        layer.swap(rx, tx);
        Ok(())
    }
}
