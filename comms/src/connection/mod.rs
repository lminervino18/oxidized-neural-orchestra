mod acceptor;
mod connector;

use crate::{
    handles::{OrchHandle, ParamServerHandle, WorkerHandle},
    transport::TransportLayer,
};

pub use acceptor::Acceptor;
pub use connector::Connector;

/// The different types of connections.
pub enum Connection<T: TransportLayer> {
    Worker(WorkerHandle<T>),
    ParamServer(ParamServerHandle<T>),
    Orchestrator(OrchHandle<T>),
}
