mod acceptor;
mod connector;

use std::fmt::{self, Display, Formatter};

use crate::{
    handles::{NodeHandle, OrchHandle, ParamServerHandle, WorkerHandle},
    transport::TransportLayer,
};

pub use acceptor::Acceptor;
pub use connector::Connector;

/// The different connection types.
#[allow(clippy::large_enum_variant)]
pub enum Connection<T>
where
    T: TransportLayer,
{
    Node(NodeHandle<T>),
    Worker(WorkerHandle<T>),
    ParamServer(ParamServerHandle<T>),
    Orch(OrchHandle<T>),
}

impl<T> Display for Connection<T>
where
    T: TransportLayer,
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
