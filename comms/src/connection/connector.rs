use std::{io, marker::PhantomData};

use tokio::io::{AsyncRead, AsyncWrite};

use crate::{
    WorkerHandle,
    handles::{NodeHandle, OrchHandle, ParamServerHandle},
    protocol::{Command, Entity, Msg},
    transport::TransportLayer,
};

/// Establishes connections and yields reliable transports.
#[derive(Debug)]
pub struct Connector<R, W, T, F>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
    T: TransportLayer,
    F: Fn(R, W) -> T,
{
    transport_factory: F,
    _phantom: PhantomData<(R, W, T)>,
}

impl<R, W, T, F> Clone for Connector<R, W, T, F>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
    T: TransportLayer,
    F: Fn(R, W) -> T + Clone,
{
    fn clone(&self) -> Self {
        Self {
            transport_factory: self.transport_factory.clone(),
            _phantom: self._phantom.clone(),
        }
    }
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
    ///
    /// # Returns
    /// A new `Connector` instance.
    pub fn new(transport_factory: F) -> Self {
        Self {
            transport_factory,
            _phantom: Default::default(),
        }
    }

    /// Connects to an uninitialised node and returns a handle to bootstrap it.
    ///
    /// The caller assigns the node's role by calling [`NodeHandle::create_server`] or
    /// [`NodeHandle::create_worker`] on the returned handle.
    ///
    /// # Args
    /// * `id` - The id number of the node.
    /// * `reader` - The reading end of the communication.
    /// * `writer` - The writing end of the communication.
    /// * `src_entity` - The entity initiating the connection.
    ///
    /// # Returns
    /// A [`NodeHandle`] ready for role assignment.
    ///
    /// # Errors
    /// Returns an io error if the connection handshake fails.
    pub async fn connect_node(
        &self,
        id: usize,
        reader: R,
        writer: W,
        src_entity: Entity,
    ) -> io::Result<NodeHandle<T>>
    where
        R: AsyncRead + Unpin,
        W: AsyncWrite + Unpin,
    {
        let transport_layer = self.connect(reader, writer, src_entity).await?;
        Ok(NodeHandle::new(id, transport_layer))
    }

    /// Connects to a worker and returns a handle to communicate with it.
    ///
    /// # Args
    /// * `id` - The id number of the server.
    /// * `reader` - The reading end of the communication.
    /// * `writer` - The writing end of the communication.
    /// * `src_entity` - The entity initiating the connection.
    ///
    /// # Returns
    /// A [`ParamServerHandle`] ready to start training.
    ///
    /// # Errors
    /// Returns an io error if the connection handshake fails.
    pub async fn connect_worker(
        &self,
        id: usize,
        reader: R,
        writer: W,
        src_entity: Entity,
    ) -> io::Result<WorkerHandle<T>>
    where
        R: AsyncRead + Unpin,
        W: AsyncWrite + Unpin,
    {
        let transport_layer = self.connect(reader, writer, src_entity).await?;
        Ok(WorkerHandle::new(id, transport_layer))
    }

    /// Connects to a parameter server and returns a handle to communicate with it.
    ///
    /// # Args
    /// * `id` - The id number of the server.
    /// * `reader` - The reading end of the communication.
    /// * `writer` - The writing end of the communication.
    /// * `src_entity` - The entity initiating the connection.
    ///
    /// # Returns
    /// A [`ParamServerHandle`] ready to exchange parameters.
    ///
    /// # Errors
    /// Returns an io error if the connection handshake fails.
    pub async fn connect_parameter_server(
        &self,
        id: usize,
        reader: R,
        writer: W,
        src_entity: Entity,
    ) -> io::Result<ParamServerHandle<T>>
    where
        R: AsyncRead + Unpin,
        W: AsyncWrite + Unpin,
    {
        let transport_layer = self.connect(reader, writer, src_entity).await?;
        Ok(ParamServerHandle::new(id, transport_layer))
    }

    /// Connects to an orchestrator and returns a handle to communicate with it.
    ///
    /// # Args
    /// * `reader` - The reading end of the communication.
    /// * `writer` - The writing end of the communication.
    /// * `src_entity` - The entity initiating the connection.
    ///
    /// # Returns
    /// An [`OrchHandle`] ready to receive events.
    ///
    /// # Errors
    /// Returns an io error if the connection handshake fails.
    pub async fn connect_orchestrator(
        &self,
        reader: R,
        writer: W,
        src_entity: Entity,
    ) -> io::Result<OrchHandle<T>>
    where
        R: AsyncRead + Unpin,
        W: AsyncWrite + Unpin,
    {
        let transport_layer = self.connect(reader, writer, src_entity).await?;
        Ok(OrchHandle::new(transport_layer))
    }

    /// Connects the given channel to an entity using a reliable transport protocol layer.
    ///
    /// # Args
    /// * `reader` - The reading end of the communication.
    /// * `writer` - The writing end of the communication.
    /// * `src_entity` - The entity initiating the connection.
    ///
    /// # Returns
    /// A new `ReliableTransport` or an io error if occurred.
    async fn connect(&self, reader: R, writer: W, src_entity: Entity) -> io::Result<T>
    where
        R: AsyncRead + Unpin,
        W: AsyncWrite + Unpin,
    {
        let mut transport_layer = (self.transport_factory)(reader, writer);
        let msg = Msg::Control(Command::Connect(src_entity));
        transport_layer.send(&msg).await?;
        Ok(transport_layer)
    }
}
