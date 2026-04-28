use std::io;

use super::{ParamServerHandle, WorkerHandle};
use crate::{
    protocol::{Command, Msg, NodeSpec, specs::{server::ServerSpec, worker::WorkerSpec}},
    transport::TransportLayer,
};

/// A handle to a node that has not yet been bootstrapped.
///
/// Obtained from [`crate::Connector::connect_node`]. Consumed when the node is assigned
/// its role via [`create_server`] or [`create_worker`].
pub struct NodeHandle<T: TransportLayer> {
    id: usize,
    transport: T,
}

impl<T: TransportLayer> NodeHandle<T> {
    pub(crate) fn new(id: usize, transport: T) -> Self {
        Self { id, transport }
    }

    /// Bootstraps the node as a parameter server and waits for session confirmation.
    ///
    /// Returns the ready handle and the session ID assigned by the node.
    pub async fn create_server(mut self, spec: ServerSpec) -> io::Result<(ParamServerHandle<T>, u64)> {
        let msg = Msg::Control(Command::CreateNode(NodeSpec::Server(spec)));
        self.transport.send(&msg).await?;

        let msg = self.transport.recv().await?;
        let Msg::Control(Command::SessionReady { session_id }) = msg else {
            return Err(io::Error::other(format!(
                "expected SessionReady from node {}, got: {msg:?}",
                self.id
            )));
        };

        Ok((ParamServerHandle::new(self.id, self.transport), session_id))
    }

    /// Bootstraps the node as a worker and returns its handle.
    pub async fn create_worker(mut self, spec: WorkerSpec) -> io::Result<WorkerHandle<T>> {
        let msg = Msg::Control(Command::CreateNode(NodeSpec::Worker(spec)));
        self.transport.send(&msg).await?;
        Ok(WorkerHandle::new(self.id, self.transport))
    }
}
