mod acceptor;
mod connector;

use crate::{
    handles::{OrchHandle, ParamServerHandle, WorkerHandle},
    transport::TransportLayer,
};

pub use acceptor::Acceptor;
pub use connector::Connector;

/// The different connection types.
#[allow(clippy::large_enum_variant)]
pub enum Connection<T: TransportLayer> {
    Worker(WorkerHandle<T>),
    ParamServer(ParamServerHandle<T>),
    Orchestrator(OrchHandle<T>),
}
