use std::{io, marker::PhantomData};

use uuid::Uuid;

use super::Connection;
use crate::{
    NodeHandle, OrchHandle, ParamServerHandle, WorkerHandle,
    protocol::{Command, Entity, Msg},
    transport::TransportLayer,
};

/// Accepts new incoming connections and assigns yields their handle types.
pub struct Acceptor<T, F>
where
    T: TransportLayer,
    F: AsyncFn() -> io::Result<T>,
{
    id: Uuid,
    transport_factory: F,
    _phantom: PhantomData<T>,
}

impl<T, F> Acceptor<T, F>
where
    T: TransportLayer,
    F: AsyncFn() -> io::Result<T>,
{
    /// Creates a new `Acceptor`.
    ///
    /// # Args
    /// * `id` - This node's id.
    /// * `transport_factory` - A transport layer factory closure.
    ///
    /// # Returns
    /// A new `Acceptor` instance.
    pub fn new(id: Uuid, transport_factory: F) -> Self {
        Self {
            id,
            transport_factory,
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
        let mut transport_layer = (self.transport_factory)().await?;

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
