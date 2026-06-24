use std::{
    fmt::{self, Debug, Formatter},
    io,
    sync::Arc,
};

use tokio::net::{
    TcpStream,
    tcp::{OwnedReadHalf, OwnedWriteHalf},
};
use uuid::Uuid;

use crate::{
    RecTP, Recon, RelTP, WorkerHandle,
    handles::{NodeHandle, OrchHandle, ParamServerHandle},
    protocol::{Command, Entity, Msg},
    transport::TransportLayer,
};

type R = OwnedReadHalf;
type W = OwnedWriteHalf;

/// Establishes connections and yields reliable transports.
#[derive(Clone)]
pub struct Connector {
    id: Uuid,
    transport_factory: Arc<dyn Fn(R, W) -> RelTP<R, W> + Send + Sync + 'static>,
}

impl Debug for Connector {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Connector").field("id", &self.id).finish()
    }
}

impl Connector {
    /// Creates a new `Connector`.
    ///
    /// # Args
    /// * `id` - This node's id.
    ///
    /// # Returns
    /// A new `Connector` instance.
    pub fn new<F>(id: Uuid, transport_factory: F) -> Self
    where
        F: Fn(R, W) -> RelTP<R, W> + Send + Sync + 'static,
    {
        Self {
            id,
            transport_factory: Arc::new(transport_factory),
        }
    }

    /// Connects to an uninitialised node and returns a handle to bootstrap it.
    ///
    /// The caller assigns the node's role by calling `NodeHandle::create_server` or
    /// `NodeHandle::create_worker` on the returned handle.
    ///
    /// # Args
    /// * `addr` - The peer's network address.
    /// * `src` - The entity initiating the connection.
    ///
    /// # Returns
    /// A new `NodeHandle` or an io error if occurred.
    pub async fn connect_node(
        &self,
        addr: &str,
        src: Entity,
    ) -> io::Result<NodeHandle<R, W, RecTP<R, W>>> {
        let (id, layer, dst) = self.connect(addr, src).await?;

        match dst {
            Entity::Node => Ok(NodeHandle::new(id, layer)),
            _ => {
                let details = format!("Invalid connection type, expected Node, got {dst:?}");
                Err(io::Error::other(details))
            }
        }
    }

    /// Connects to a worker and returns a handle to communicate with it.
    ///
    /// # Args
    /// * `addr` - The peer's network address.
    /// * `src` - The entity initiating the connection.
    ///
    /// # Returns
    /// A `WorkerHandle` ready to start training.
    pub async fn connect_worker(
        &self,
        addr: &str,
        src: Entity,
    ) -> io::Result<WorkerHandle<R, W, RecTP<R, W>>> {
        let (id, layer, dst) = self.connect(addr, src).await?;

        match dst {
            Entity::Worker => Ok(WorkerHandle::new(id, layer)),
            _ => {
                let details = format!("Invalid connection type, expected Worker, got {dst:?}");
                Err(io::Error::other(details))
            }
        }
    }

    /// Connects to a parameter server and returns a handle to communicate with it.
    ///
    /// # Args
    /// * `addr` - The peer's network address.
    /// * `src` - The entity initiating the connection.
    ///
    /// # Returns
    /// A new `ParamServerHandle` or an io error if occurred.
    pub async fn connect_parameter_server(
        &self,
        addr: &str,
        src: Entity,
    ) -> io::Result<ParamServerHandle<R, W, RecTP<R, W>>> {
        let (id, layer, dst) = self.connect(addr, src).await?;

        match dst {
            Entity::ParamServer => Ok(ParamServerHandle::new(id, layer)),
            _ => {
                let details = format!("Invalid connection type, expected Server, got {dst:?}");
                Err(io::Error::other(details))
            }
        }
    }

    /// Connects to an orchestrator and returns a handle to communicate with it.
    ///
    /// # Args
    /// * `addr` - The peer's network address.
    /// * `src` - The entity initiating the connection.
    ///
    /// # Returns
    /// A new `OrchHandle` or an io error if occurred.
    pub async fn connect_orchestrator(
        &self,
        addr: &str,
        src: Entity,
    ) -> io::Result<OrchHandle<R, W, RecTP<R, W>>> {
        let (id, layer, dst) = self.connect(addr, src).await?;

        match dst {
            Entity::Orchestrator => Ok(OrchHandle::new(id, layer)),
            _ => {
                let details = format!("Invalid connection type, expected Orch, got {dst:?}");
                Err(io::Error::other(details))
            }
        }
    }

    /// Reconnects the given transport layer with the new reader and writer.
    ///
    /// # Args
    /// * `addr` - The peer's network address.
    /// * `layer` - The transport layer used to communicate with the other end.
    ///
    /// # Returns
    /// An io error if occurred.
    pub async fn reconnect<T>(&self, addr: &str, layer: &mut T) -> io::Result<()>
    where
        T: TransportLayer<R, W>,
    {
        let (rx, tx) = self.connect_io(addr).await?;
        layer.swap(rx, tx);
        self.handshake(Entity::Reconnect, layer).await.map(|_| ())
    }

    /// Connects the given channel to an entity using a reliable transport protocol layer.
    ///
    /// # Args
    /// * `addr` - The peer's network address.
    /// * `src` - The entity initiating the connection.
    ///
    /// # Returns
    /// The peer's id and entity, and the transport layer to communicate or an io error if occurred.
    async fn connect(&self, addr: &str, src: Entity) -> io::Result<(Uuid, RecTP<R, W>, Entity)> {
        let (rx, tx) = self.connect_io(addr).await?;
        let mut layer = (self.transport_factory)(rx, tx);
        let (id, dst) = self.handshake(src, &mut layer).await?;
        let recon_layer = Recon::active(addr.to_string(), self.clone(), layer);
        Ok((id, recon_layer, dst))
    }

    /// Makes a connection with the peer and yields the inner reader and writer for that connection.
    ///
    /// # Args
    /// * `addr` - The peer's network address.
    ///
    /// # Returns
    /// Both socket ends or an io error if occurred.
    async fn connect_io(&self, addr: &str) -> io::Result<(R, W)> {
        let stream = TcpStream::connect(addr).await?;
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
        let msg = Msg::Control(Command::Connect { id: self.id, src });
        layer.send(&msg).await?;

        let msg = layer.recv().await?;
        let Msg::Control(Command::Accept { id, src: dst }) = msg else {
            let details = format!("Invalid connection message, expected Accept, got {msg:?}");
            return Err(io::Error::other(details));
        };

        Ok((id, dst))
    }
}
