use std::{io, marker::PhantomData};

use tokio::io::{AsyncRead, AsyncWrite};
use uuid::Uuid;

use crate::{
    Connection, WorkerHandle,
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
    id: Uuid,
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
            id: self.id,
            transport_factory: self.transport_factory.clone(),
            _phantom: self._phantom,
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
    /// * `rx` - The reading end of the communication.
    /// * `tx` - The writing end of the communication.
    /// * `src` - The entity initiating the connection.
    ///
    /// # Returns
    /// A new `NodeHandle` or an io error if occurred.
    pub async fn connect_node(&self, rx: R, tx: W, src: Entity) -> io::Result<NodeHandle<T>>
    where
        R: AsyncRead + Unpin,
        W: AsyncWrite + Unpin,
    {
        match self.connect(rx, tx, src).await? {
            Connection::Node(node_handle) => Ok(node_handle),
            conn => {
                let details = format!("Invalid connection type, expected Node, got {conn}");
                Err(io::Error::other(details))
            }
        }
    }

    /// Connects to a worker and returns a handle to communicate with it.
    ///
    /// # Args
    /// * `rx` - The reading end of the communication.
    /// * `tx` - The writing end of the communication.
    /// * `src` - The entity initiating the connection.
    ///
    /// # Returns
    /// A `WorkerHandle` ready to start training.
    pub async fn connect_worker(&self, rx: R, tx: W, src: Entity) -> io::Result<WorkerHandle<T>>
    where
        R: AsyncRead + Unpin,
        W: AsyncWrite + Unpin,
    {
        match self.connect(rx, tx, src).await? {
            Connection::Worker(worker_handle) => Ok(worker_handle),
            conn => {
                let details = format!("Invalid connection type, expected Worker, got {conn}");
                Err(io::Error::other(details))
            }
        }
    }

    /// Connects to a parameter server and returns a handle to communicate with it.
    ///
    /// # Args
    /// * `rx` - The reading end of the communication.
    /// * `tx` - The writing end of the communication.
    /// * `src` - The entity initiating the connection.
    ///
    /// # Returns
    /// A new `ParamServerHandle` or an io error if occurred.
    pub async fn connect_parameter_server(
        &self,
        rx: R,
        tx: W,
        src: Entity,
    ) -> io::Result<ParamServerHandle<T>>
    where
        R: AsyncRead + Unpin,
        W: AsyncWrite + Unpin,
    {
        match self.connect(rx, tx, src).await? {
            Connection::ParamServer(server_handle) => Ok(server_handle),
            conn => {
                let details = format!("Invalid connection type, expected Server, got {conn}");
                Err(io::Error::other(details))
            }
        }
    }

    /// Connects to an orchestrator and returns a handle to communicate with it.
    ///
    /// # Args
    /// * `rx` - The reading end of the communication.
    /// * `tx` - The writing end of the communication.
    /// * `src` - The entity initiating the connection.
    ///
    /// # Returns
    /// A new `OrchHandle` or an io error if occurred.
    pub async fn connect_orchestrator(&self, rx: R, tx: W, src: Entity) -> io::Result<OrchHandle<T>>
    where
        R: AsyncRead + Unpin,
        W: AsyncWrite + Unpin,
    {
        match self.connect(rx, tx, src).await? {
            Connection::Orch(orch_handle) => Ok(orch_handle),
            conn => {
                let details = format!("Invalid connection type, expected Orchestrator, got {conn}");
                Err(io::Error::other(details))
            }
        }
    }

    /// Connects the given channel to an entity using a reliable transport protocol layer.
    ///
    /// # Args
    /// * `rx` - The reading end of the communication.
    /// * `tx` - The writing end of the communication.
    /// * `src` - The entity initiating the connection.
    ///
    /// # Returns
    /// A new `TransportLayer` or an io error if occurred.
    async fn connect(&self, rx: R, tx: W, src: Entity) -> io::Result<Connection<T>> {
        let mut stack = (self.transport_factory)(rx, tx);
        let msg = Msg::Control(Command::Connect { id: self.id, src });
        stack.send(&msg).await?;

        let msg = stack.recv().await?;
        let Msg::Control(Command::Accept { id, src: dst }) = msg else {
            let details = format!("Invalid connection message, expected Accept, got {msg:?}");
            return Err(io::Error::other(details));
        };

        let conn = match dst {
            Entity::Node => Connection::Node(NodeHandle::new(id, stack)),
            Entity::Orchestrator => Connection::Orch(OrchHandle::new(id, stack)),
            Entity::ParamServer => Connection::ParamServer(ParamServerHandle::new(id, stack)),
            Entity::Worker => Connection::Worker(WorkerHandle::new(id, stack)),
        };

        Ok(conn)
    }
}
