use std::{io, marker::PhantomData};

use tokio::io::{AsyncRead, AsyncWrite};

use crate::{
    handles::{OrchHandle, ParamServerHandle, WorkerHandle},
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

    /// Connects the given channel to a worker using a reliable transport protocol layer.
    ///
    /// # Args
    /// * `id` - The id of the server.
    /// * `reader` - The reading end of the communication.
    /// * `writer` - The writing end of the communication.
    ///
    /// # Returns
    /// A new `WorkerHandle` instance.
    pub async fn connect_worker(
        &self,
        id: usize,
        reader: R,
        writer: W,
    ) -> io::Result<WorkerHandle<T>>
    where
        R: AsyncRead + Unpin,
        W: AsyncWrite + Unpin,
    {
        let transport_layer = self.connect(reader, writer).await?;
        Ok(WorkerHandle::new(id, transport_layer))
    }

    /// Connects the given channel to a parameter server using a reliable transport protocol layer.
    ///
    /// # Args
    /// * `id` - The id of the worker.
    /// * `reader` - The reading end of the communication.
    /// * `writer` - The writing end of the communication.
    ///
    /// # Returns
    /// A new `ParamServerHandle` instance.
    pub async fn connect_parameter_server(
        &self,
        id: usize,
        reader: R,
        writer: W,
    ) -> io::Result<ParamServerHandle<T>>
    where
        R: AsyncRead + Unpin,
        W: AsyncWrite + Unpin,
    {
        let transport_layer = self.connect(reader, writer).await?;
        Ok(ParamServerHandle::new(id, transport_layer))
    }

    /// Connects the given channel to an orchestrator using a reliable transport protocol layer.
    ///
    /// # Args
    /// * `reader` - The reading end of the communication.
    /// * `writer` - The writing end of the communication.
    ///
    /// # Returns
    /// A new `OrchHandle` instance.
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
