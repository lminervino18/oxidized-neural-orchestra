use std::{io, marker::PhantomData};

use tokio::{
    io::{AsyncRead, AsyncWrite},
    net::{
        TcpStream,
        tcp::{OwnedReadHalf, OwnedWriteHalf},
    },
};
use uuid::Uuid;

use crate::{
    WorkerHandle,
    handles::{NodeHandle, OrchHandle, ParamServerHandle},
    protocol::{Command, Entity, Msg},
    transport::{IoSwapable, TransportLayer},
};

type R = OwnedReadHalf;
type W = OwnedWriteHalf;

/// Establishes connections and yields reliable transports.
#[derive(Debug)]
pub struct Connector<T, F>
where
    T: TransportLayer,
    F: Fn(R, W) -> T,
{
    id: Uuid,
    transport_factory: F,
    _phantom: PhantomData<(R, W, T)>,
}

impl<T, F> Clone for Connector<T, F>
where
    T: TransportLayer,
    F: Fn(R, W) -> T + Clone,
{
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            transport_factory: self.transport_factory.clone(),
            _phantom: self._phantom,
        }
    }
}

impl<T, F> Connector<T, F>
where
    T: TransportLayer,
    F: Fn(R, W) -> T,
{
    /// Creates a new `Connector`.
    ///
    /// # Args
    /// * `id` - This node's id.
    /// * `transport_factory` - A factory of transport layers.
    ///
    /// # Returns
    /// A new `Connector` instance.
    pub fn new(id: Uuid, transport_factory: F) -> Self {
        Self {
            id,
            transport_factory,
            _phantom: Default::default(),
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
    pub async fn connect_node(&self, addr: &str, src: Entity) -> io::Result<NodeHandle<T>>
    where
        R: AsyncRead + Unpin,
        W: AsyncWrite + Unpin,
    {
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
    pub async fn connect_worker(&self, addr: &str, src: Entity) -> io::Result<WorkerHandle<T>>
    where
        R: AsyncRead + Unpin,
        W: AsyncWrite + Unpin,
    {
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
    ) -> io::Result<ParamServerHandle<T>>
    where
        R: AsyncRead + Unpin,
        W: AsyncWrite + Unpin,
    {
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
    pub async fn connect_orchestrator(&self, addr: &str, src: Entity) -> io::Result<OrchHandle<T>>
    where
        R: AsyncRead + Unpin,
        W: AsyncWrite + Unpin,
    {
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
    pub async fn reconnect<U>(&self, addr: &str, layer: &mut U) -> io::Result<()>
    where
        U: TransportLayer + IoSwapable<R, W>,
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
    async fn connect(&self, addr: &str, src: Entity) -> io::Result<(Uuid, T, Entity)> {
        let (rx, tx) = self.connect_io(addr).await?;
        let mut layer = (self.transport_factory)(rx, tx);
        let (id, dst) = self.handshake(src, &mut layer).await?;
        Ok((id, layer, dst))
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
    async fn handshake<U>(&self, src: Entity, layer: &mut U) -> io::Result<(Uuid, Entity)>
    where
        U: TransportLayer,
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
