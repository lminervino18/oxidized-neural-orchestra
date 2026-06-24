use std::{
    collections::HashMap,
    fmt::{self, Debug, Formatter},
    io,
    sync::Arc,
};

use log::info;
use tokio::{
    net::{
        TcpListener,
        tcp::{OwnedReadHalf, OwnedWriteHalf},
    },
    sync::{Mutex, Notify},
};
use uuid::Uuid;

use super::Connection;
use crate::{
    NodeHandle, OrchHandle, ParamServerHandle, RecTP, Recon, RelTP, WorkerHandle,
    protocol::{Command, Entity, Msg},
    transport::TransportLayer,
};

type R = OwnedReadHalf;
type W = OwnedWriteHalf;

/// The shared state for managing incoming connections.
#[derive(Debug)]
struct ConnRegistry {
    backlog: Mutex<HashMap<Uuid, (R, W)>>,
    listener: TcpListener,
    notify: Notify,
}

/// Accepts new incoming connections and assigns yields their handle types.
#[derive(Clone)]
pub struct Acceptor {
    id: Uuid,
    conns: Arc<ConnRegistry>,
    transport_factory: Arc<dyn Fn(R, W) -> RelTP<R, W> + Send + Sync + 'static>,
}

impl Debug for Acceptor {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Acceptor")
            .field("id", &self.id)
            .field("conns", &self.conns)
            .finish()
    }
}

impl Acceptor {
    /// Creates a new `Acceptor`.
    ///
    /// # Args
    /// * `id` - This node's id.
    /// * `listener` - A listener for incoming connections.
    /// * `transport_factory` - A clossure to build a reliable transport layer.
    ///
    /// # Returns
    /// A new `Acceptor` instance.
    pub fn new<F>(id: Uuid, listener: TcpListener, transport_factory: F) -> Self
    where
        F: Fn(R, W) -> RelTP<R, W> + Send + Sync + 'static,
    {
        let conn_registry = ConnRegistry {
            backlog: Mutex::new(HashMap::new()),
            listener,
            notify: Notify::new(),
        };

        Self {
            id,
            conns: Arc::new(conn_registry),
            transport_factory: Arc::new(transport_factory),
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
    pub async fn accept(&self, src: Entity) -> io::Result<Connection<R, W, RecTP<R, W>>> {
        let (rx, tx) = self.accept_connection().await?;
        let mut layer = (self.transport_factory)(rx, tx);
        let (id, dst) = self.handshake(src, &mut layer).await?;
        let recon_layer = Recon::passive(self.id, self.clone(), layer);

        let conn = match dst {
            Entity::Worker => Connection::Worker(WorkerHandle::new(id, recon_layer)),
            Entity::ParamServer => Connection::ParamServer(ParamServerHandle::new(id, recon_layer)),
            Entity::Orchestrator => Connection::Orch(OrchHandle::new(id, recon_layer)),
            Entity::Node => Connection::Node(NodeHandle::new(id, recon_layer)),
            entity => {
                let details = format!("Accepted invalid incoming connection from {entity:?}");
                return Err(io::Error::other(details));
            }
        };

        Ok(conn)
    }

    /// Waits for a new connection through the inner listener.
    ///
    /// # Returns
    /// A splitted socket or an io error if occurred.
    async fn accept_connection(&self) -> io::Result<(R, W)> {
        let (stream, addr) = self.conns.listener.accept().await?;
        info!("new incoming connection from {addr}");
        Ok(stream.into_split())
    }

    /// Makes the handshake with the peer sharing the entity types and ids.
    ///
    /// # Args
    /// * `src` - This node's entity variant.
    /// * `layer` - The transport layer used to communicate with the other end.
    ///
    /// # Returns
    /// The peer's id and entity type or an io error if occurred.
    async fn handshake<T>(&self, src: Entity, layer: &mut T) -> io::Result<(Uuid, Entity)>
    where
        T: TransportLayer<R, W>,
    {
        let msg = layer.recv().await?;

        let Msg::Control(Command::Connect { id, src: dst }) = msg else {
            let text = format!("Expected Connect message, got: {msg:?}");
            return Err(io::Error::other(text));
        };

        let msg = Msg::Control(Command::Accept { id: self.id, src });
        layer.send(&msg).await?;

        Ok((id, dst))
    }

    /// Will wait until a connection with the expected `id` arrives blocking the thread in the process.
    ///
    /// # Args
    /// * `id` - The id that's being waited for.
    /// * `layer` - The transport layer used to communicate with the other end.
    ///
    /// # Returns
    /// An io error if occurred.
    pub async fn reconnect<T>(&self, id: Uuid, layer: &mut T) -> io::Result<()>
    where
        T: TransportLayer<R, W>,
    {
        let (rx, tx) = loop {
            {
                let mut backlog = self.conns.backlog.lock().await;

                if let Some(rx_tx) = backlog.remove(&id) {
                    break rx_tx;
                }
            }

            let (rx, tx) = tokio::select! {
                _ = self.conns.notify.notified() => continue,
                res = self.accept_connection() => res?,
            };

            let mut tmp = (self.transport_factory)(rx, tx);
            let (peer_id, dst) = self.handshake(Entity::Reconnect, &mut tmp).await?;

            if dst != Entity::Reconnect {
                let details = format!("Expected Entity::Reconnect, got: {dst:?} with id {peer_id}");
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
