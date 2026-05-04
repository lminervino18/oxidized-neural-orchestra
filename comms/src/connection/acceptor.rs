use std::{io, marker::PhantomData};

use super::Connection;
use crate::{
    OrchHandle, ParamServerHandle, WorkerHandle,
    protocol::{Command, Entity, Msg},
    transport::TransportLayer,
};

/// Accepts new incoming connections and assigns yields their handle types.
pub struct Acceptor<T, F>
where
    T: TransportLayer,
    F: AsyncFn() -> io::Result<T>,
{
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
    /// * `transport_factory` - A transport layer factory closure.
    ///
    /// # Returns
    /// A new `Acceptor` instance.
    pub fn new(transport_factory: F) -> Self {
        Self {
            transport_factory,
            _phantom: Default::default(),
        }
    }

    /// Blocks the current thread until a new connection arrives.
    ///
    /// # Returns
    /// A new connection or an io error if occurred while waiting for incoming connections
    /// or receiving the type of entity from the peer.
    pub async fn accept(&mut self) -> io::Result<Connection<T>> {
        let mut transport_layer = (self.transport_factory)().await?;

        let msg = transport_layer.recv().await?;
        let Msg::Control(Command::Connect(entity)) = msg else {
            let text = format!("Expected Connect message, got: {msg:?}");
            return Err(io::Error::other(text));
        };

        let conn = match entity {
            Entity::Worker { id } => Connection::Worker(WorkerHandle::new(id, transport_layer)),
            Entity::ParamServer { id } => {
                Connection::ParamServer(ParamServerHandle::new(id, transport_layer))
            }
            Entity::Orchestrator => Connection::Orchestrator(OrchHandle::new(transport_layer)),
        };

        Ok(conn)
    }
}
