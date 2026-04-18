mod codec;
mod connection;
mod handles;
pub mod protocol;
mod share_dataset;
mod sparse;
mod transport;
mod utils;

use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};

pub use connection::{Acceptor, Connection, Connector};
pub use handles::{
    OrchHandle, ParamServerHandle, PullParamsResponse, PullSpecResponse, WorkerEvent, WorkerHandle,
};
pub use protocol::specs;
pub use sparse::Float01;
pub use transport::{Rtp, Stp, TransportLayer, build_reliable_transport, build_simple_transport};

/// The network TCP reliable transport layer.
pub type NetRtp = Rtp<OwnedReadHalf, OwnedWriteHalf>;
