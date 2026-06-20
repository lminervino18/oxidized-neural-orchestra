use std::{io, marker::PhantomData};

use tokio::io::{AsyncRead, AsyncWrite};
use uuid::Uuid;

use super::Connection;
use crate::{
    NodeHandle, OrchHandle, ParamServerHandle, WorkerHandle,
    protocol::{Command, Entity, Msg},
    transport::TransportLayer,
};

/// Accepts new incoming connections and assigns yields their handle types.
pub struct Acceptor<R, W, T, F, G>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
    T: TransportLayer,
    F: Fn(R, W) -> T,
    G: AsyncFn() -> io::Result<(R, W)>,
{
    id: Uuid,
    transport_factory: F,
    connection_factory: G,
    _phantom: PhantomData<(T, G)>,
}

impl<R, W, T, F, G> Acceptor<R, W, T, F, G>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
    T: TransportLayer,
    F: Fn(R, W) -> T,
    G: AsyncFn() -> io::Result<(R, W)>,
{
    /// Creates a new `Acceptor`.
    ///
    /// # Args
    /// * `id` - This node's id.
    /// * `connection_factory` - A closure that waits for new connections.
    /// * `transport_factory` - A transport layer factory closure.
    ///
    /// # Returns
    /// A new `Acceptor` instance.
    pub fn new(id: Uuid, transport_factory: F, connection_factory: G) -> Self {
        Self {
            id,
            transport_factory,
            connection_factory,
            _phantom: Default::default(),
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
    pub async fn accept(&mut self, src: Entity) -> io::Result<Connection<T>> {
        let (rx, tx) = (self.connection_factory)().await?;
        let mut transport_layer = (self.transport_factory)(rx, tx);

        let msg = transport_layer.recv().await?;
        let Msg::Control(Command::Connect { id, src: dst }) = msg else {
            let text = format!("Expected Connect message, got: {msg:?}");
            return Err(io::Error::other(text));
        };

        let msg = Msg::Control(Command::Accept { id: self.id, src });
        transport_layer.send(&msg).await?;

        let conn = match dst {
            Entity::Worker => Connection::Worker(WorkerHandle::new(id, transport_layer)),
            Entity::ParamServer => {
                Connection::ParamServer(ParamServerHandle::new(id, transport_layer))
            }
            Entity::Orchestrator => Connection::Orch(OrchHandle::new(id, transport_layer)),
            Entity::Node => Connection::Node(NodeHandle::new(id, transport_layer)),
        };

        Ok(conn)
    }
}
