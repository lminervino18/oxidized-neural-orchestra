use std::{
    io,
    time::{Duration, Instant},
};

use super::{ParamServerHandle, WorkerHandle};
use crate::{
    protocol::{
        Command, Msg,
        specs::{node::NodeSpec, server::ServerSpec, worker::WorkerSpec},
    },
    transport::TransportLayer,
};

/// A handle to a node that has not yet been bootstrapped.
///
/// Obtained from [`crate::Connector::connect_node`]. Consumed when the node is assigned
/// its role via [`NodeHandle::create_server`] or [`NodeHandle::create_worker`].
pub struct NodeHandle<T: TransportLayer> {
    id: usize,
    transport: T,
}

impl<T: TransportLayer> NodeHandle<T> {
    /// Creates a new `NodeHandle`.
    ///
    /// # Args
    /// * `id` - The id number of the node.
    /// * `transport` - The transport layer of the communication.
    ///
    /// # Returns
    /// A new `NodeHandle` instance.
    pub(crate) fn new(id: usize, transport: T) -> Self {
        Self { id, transport }
    }

    /// Bootstraps the node as a parameter server.
    ///
    /// # Args
    /// * `spec` - The specification for the parameter server.
    ///
    /// # Returns
    /// The parameter server handle or an io error if occurred.
    pub async fn create_server(mut self, spec: ServerSpec) -> io::Result<ParamServerHandle<T>> {
        self.create(NodeSpec::Server(spec)).await?;
        Ok(ParamServerHandle::new(self.id, self.transport))
    }

    /// Bootstraps the node as a worker and returns its handle.
    ///
    /// # Args
    /// * `spec` - The specification for the worker.
    ///
    /// # Returns
    /// The ready worker handle or an io error if occurred.
    pub async fn create_worker(mut self, spec: WorkerSpec) -> io::Result<WorkerHandle<T>> {
        self.create(NodeSpec::Worker(spec)).await?;
        Ok(WorkerHandle::new(self.id, self.transport))
    }

    /// Sends a create message to the other end with the given specification.
    ///
    /// # Args
    /// * `spec` - The specification for the node.
    ///
    /// # Returns
    /// An io error if occurred.
    async fn create(&mut self, spec: NodeSpec) -> io::Result<()> {
        let msg = Msg::Control(Command::CreateNode { spec });
        self.transport.send(&msg).await
    }

    /// Sends a ping request to the node.
    ///
    /// # Returns
    /// The round trip time duration the other node took to respond.
    pub async fn ping(&mut self) -> io::Result<Duration> {
        let msg = Msg::Control(Command::Ping);

        let start = Instant::now();
        self.transport.send(&msg).await?;
        Ok(start.elapsed())
    }

    /// Responds to a ping request by sending a pong response.
    ///
    /// # Returns
    /// An io error if occurred.
    pub async fn pong(&mut self) -> io::Result<()> {
        let msg = Msg::Control(Command::Pong);
        self.transport.send(&msg).await
    }

    /// Disconncts the node.
    ///
    /// # Returns
    /// An io error if occurred.
    pub async fn disconnect(&mut self) -> io::Result<()> {
        let msg = Msg::Controll(Command::Disconnect);
        self.transport.send(&msg).await
    }
}
