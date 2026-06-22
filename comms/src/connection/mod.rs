mod acceptor;
mod connector;

use std::fmt::{self, Display, Formatter};

use crate::{
    handles::{NodeHandle, OrchHandle, ParamServerHandle, WorkerHandle},
    transport::TransportLayer,
};

pub use acceptor::Acceptor;
pub use connector::Connector;
use tokio::io::{AsyncRead, AsyncWrite};

/// The different connection types.
#[allow(clippy::large_enum_variant)]
pub enum Connection<R, W, T>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
    T: TransportLayer<R, W>,
{
    Node(NodeHandle<R, W, T>),
    Worker(WorkerHandle<R, W, T>),
    ParamServer(ParamServerHandle<R, W, T>),
    Orch(OrchHandle<R, W, T>),
}

impl<R, W, T> Display for Connection<R, W, T>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
    T: TransportLayer<R, W>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Node(node_handle) => format!("Node[{}]", node_handle.id()),
            Self::Worker(worker_handle) => format!("Worker[{}]", worker_handle.id()),
            Self::ParamServer(server_handle) => format!("Server[{}]", server_handle.id()),
            Self::Orch(orch_handle) => format!("Orch[{}]", orch_handle.id()),
        };

        f.write_str(&s)
    }
}
