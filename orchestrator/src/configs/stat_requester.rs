use std::{collections::BTreeMap, io};

use comms::{
    Connector, NodeHandle, TransportLayer,
    protocol::Entity,
    specs::node::{StatRequest, StatResponse},
};
use tokio::net::{
    TcpStream,
    tcp::{OwnedReadHalf, OwnedWriteHalf},
};

type R = OwnedReadHalf;
type W = OwnedWriteHalf;

/// The amount of ping rounds to make.
const PING_ROUNDS: usize = 10;

/// Obtains the statistics from the nodes in the network.
pub struct StatRequester<T, F>
where
    T: TransportLayer,
    F: Fn(R, W) -> T,
{
    connector: Connector<R, W, T, F>,
}

impl<T, F> StatRequester<T, F>
where
    T: TransportLayer,
    F: Fn(R, W) -> T,
{
    /// Creates a new `Resovler`.
    ///
    /// # Args
    /// * `connector` - The connection establisher.
    ///
    /// # Returns
    /// A new `Resolver` instance.
    pub fn new(connector: Connector<R, W, T, F>) -> Self {
        Self { connector }
    }

    /// Connects to the nodes given their addresses and calculates requests the relevant
    /// statistics for training.
    ///
    /// # Args
    /// * `addrs` - The network addresses of the workers.
    ///
    /// # Returns
    /// The accumulated statistic responses or an io error if occurred.
    pub async fn obtain_stats(
        &mut self,
        addrs: &[String],
    ) -> io::Result<BTreeMap<String, Vec<StatResponse>>> {
        let mut handles = self.connect_nodes(addrs).await?;
        self.request_stats(&mut handles, addrs).await?;
        let stats = self.wait_for_responses(&mut handles, addrs).await?;
        self.disconnect_nodes(&mut handles).await;
        Ok(stats)
    }

    /// Connects to the nodes in the network.
    ///
    /// # Args
    /// * `addrs` - The network addresses of the nodes.
    ///
    /// # Returns
    /// The `NodeHandle` for each of the given network address or an io error if occurred.
    async fn connect_nodes(&mut self, addrs: &[String]) -> io::Result<Vec<NodeHandle<T>>> {
        let mut handles = Vec::with_capacity(addrs.len());

        for (id, addr) in addrs.iter().enumerate() {
            let stream = TcpStream::connect(addr).await?;
            let (rx, tx) = stream.into_split();

            let node_handle = self
                .connector
                .connect_node(id, rx, tx, Entity::Orchestrator)
                .await?;

            handles.push(node_handle);
        }

        Ok(handles)
    }

    /// Requests the nodes to calculate their ping latency between each other.
    ///
    /// # Args
    /// * `handles` - The handles for communicating with the nodes.
    /// * `addrs` - The network addresses of each of the handles.
    ///
    /// # Returns
    /// An io error if occurred.
    async fn request_stats(
        &self,
        handles: &mut [NodeHandle<T>],
        addrs: &[String],
    ) -> io::Result<()> {
        for (id, node_handle) in handles.iter_mut().enumerate() {
            let ping_req = StatRequest::Ping {
                addrs: addrs[..id].to_vec(),
                rounds: PING_ROUNDS,
                incoming: addrs.len() - id,
            };

            node_handle.push_stats(vec![ping_req]).await?;
        }

        Ok(())
    }

    /// Waits for the statistic responses of each of the nodes.
    ///
    /// # Args
    /// * `handles` - The handles for communicating with the nodes.
    /// * `addrs` - The network addresses of each of the handles.
    ///
    /// # Returns
    /// The statistic responses or an io error if occurred.
    async fn wait_for_responses(
        &self,
        handles: &mut [NodeHandle<T>],
        addrs: &[String],
    ) -> io::Result<BTreeMap<String, Vec<StatResponse>>> {
        let mut stats = BTreeMap::new();

        for (addr, node_handle) in addrs.iter().zip(handles) {
            let node_stats = node_handle.pull_stats().await?;
            stats.insert(addr.to_string(), node_stats);
        }

        Ok(stats)
    }

    /// Disconnects the node handles from the network.
    ///
    /// # Args
    /// * `handles` - The handles for communicating with the nodes.
    async fn disconnect_nodes(&self, handles: &mut [NodeHandle<T>]) {
        for node_handle in handles {
            let _ = node_handle.disconnect().await;
        }
    }
}
