use std::{io, marker::PhantomData};

use tokio::io::{AsyncRead, AsyncWrite};

use crate::{
    handles::{NodeHandle, OrchHandle, ParamServerHandle},
    protocol::{Command, Entity, Msg},
    transport::TransportLayer,
};

/// Establishes connections and yields reliable transports.
#[derive(Debug, Clone, Copy)]
pub struct Connector<R, W, T, F>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
    T: TransportLayer,
    F: FnMut(R, W) -> T,
{
    transport_factory: F,
    src_entity: Entity,
    _phantom: PhantomData<(R, W, T)>,
}

impl<R, W, T, F> Connector<R, W, T, F>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
    T: TransportLayer,
    F: Fn(R, W) -> T,
{
    /// Creates a new `Connector`.
    ///
    /// # Args
    /// * `transport_factory` - A factory of transport layers.
    /// * `entity` - The callee's entity.
    ///
    /// # Returns
    /// A new `Connector` instance.
    pub fn new(transport_factory: F, src_entity: Entity) -> Self {
        Self {
            transport_factory,
            src_entity,
            _phantom: Default::default(),
        }
    }

    /// Connects to an uninitialised node and returns a [`NodeHandle`] to bootstrap it.
    ///
    /// The caller decides the node's role by calling [`NodeHandle::create_server`] or
    /// [`NodeHandle::create_worker`] on the returned handle.
    pub async fn connect_node(&self, id: usize, reader: R, writer: W) -> io::Result<NodeHandle<T>>
    where
        R: AsyncRead + Unpin,
        W: AsyncWrite + Unpin,
    {
        let transport_layer = self.connect(reader, writer).await?;
        Ok(NodeHandle::new(id, transport_layer))
    }

    /// Connects to a running parameter server session and sends the join message.
    ///
    /// Used by workers to attach to an already-bootstrapped session identified by `session_id`.
    pub async fn join_server_session(
        &self,
        id: usize,
        reader: R,
        writer: W,
        session_id: u64,
    ) -> io::Result<ParamServerHandle<T>>
    where
        R: AsyncRead + Unpin,
        W: AsyncWrite + Unpin,
    {
        let transport_layer = self.connect(reader, writer).await?;
        let mut handle = ParamServerHandle::new(id, transport_layer);
        handle.join_session(session_id).await?;
        Ok(handle)
    }

    /// Connects the given channel to an orchestrator using a reliable transport protocol layer.
    pub async fn connect_orchestrator(&self, reader: R, writer: W) -> io::Result<OrchHandle<T>>
    where
        R: AsyncRead + Unpin,
        W: AsyncWrite + Unpin,
    {
        let transport_layer = self.connect(reader, writer).await?;
        Ok(OrchHandle::new(transport_layer))
    }

    /// Connects the given channel to an entity using a reliable transport protocol layer.
    ///
    /// # Args
    /// * `reader` - The reading end of the communication.
    /// * `writer` - The writing end of the communication.
    ///
    /// # Returns
    /// A new `ReliableTransport` or an io error if occurred.
    async fn connect(&self, reader: R, writer: W) -> io::Result<T>
    where
        R: AsyncRead + Unpin,
        W: AsyncWrite + Unpin,
    {
        let mut transport_layer = (self.transport_factory)(reader, writer);
        let msg = Msg::Control(Command::Connect(self.src_entity));
        transport_layer.send(&msg).await?;
        Ok(transport_layer)
    }
}
