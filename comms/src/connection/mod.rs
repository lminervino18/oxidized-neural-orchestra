mod acceptor;
mod connector;

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
