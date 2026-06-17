use std::{collections::BTreeMap, io};

use comms::{
    NodeHandle, TransportLayer,
    specs::node::{StatRequest, StatResponse},
};

/// The amount of ping rounds to make.
const PING_ROUNDS: usize = 10;

/// Obtains the statistics from the nodes in the network.
pub struct StatRequester;

impl StatRequester {
    /// Creates a new `Resovler`.
    ///
    /// # Returns
    /// A new `Resolver` instance.
    pub fn new() -> Self {
        Self
    }

    /// Connects to the nodes given their addresses and calculates requests the relevant
    /// statistics for training.
    ///
    /// # Args
    /// * `handles` - The handles for communicating with the nodes.
    ///
    /// # Returns
    /// The accumulated statistic responses or an io error if occurred.
    pub async fn obtain_stats<T>(
        &mut self,
        mut handles: BTreeMap<String, NodeHandle<T>>,
    ) -> io::Result<BTreeMap<String, Vec<StatResponse>>>
    where
        T: TransportLayer,
    {
        self.request_stats(&mut handles).await?;
        let stats = self.wait_for_responses(&mut handles).await?;
        self.disconnect_nodes(handles.values_mut()).await;
        Ok(stats)
    }

    /// Requests the nodes to calculate their ping latency between each other.
    ///
    /// # Args
    /// * `handles` - The handles for communicating with the nodes.
    ///
    /// # Returns
    /// An io error if occurred.
    async fn request_stats<T>(
        &self,
        handles: &mut BTreeMap<String, NodeHandle<T>>,
    ) -> io::Result<()>
    where
        T: TransportLayer,
    {
        let addrs: Vec<_> = handles.keys().cloned().collect();

        for (i, node_handle) in handles.values_mut().enumerate() {
            let ping_req = StatRequest::Ping {
                addrs: addrs[..i].to_vec(),
                rounds: PING_ROUNDS,
                incoming: addrs.len() - i - 1,
            };

            node_handle.push_stats(vec![ping_req]).await?;
        }

        Ok(())
    }

    /// Waits for the statistic responses of each of the nodes.
    ///
    /// # Args
    /// * `handles` - The handles for communicating with the nodes.
    ///
    /// # Returns
    /// The statistic responses or an io error if occurred.
    async fn wait_for_responses<T>(
        &self,
        handles: &mut BTreeMap<String, NodeHandle<T>>,
    ) -> io::Result<BTreeMap<String, Vec<StatResponse>>>
    where
        T: TransportLayer,
    {
        let mut stats = BTreeMap::new();

        for (addr, node_handle) in handles.iter_mut() {
            let node_stats = node_handle.pull_stats().await?;
            stats.insert(addr.to_string(), node_stats);
        }

        Ok(stats)
    }

    /// Disconnects the node handles from the network.
    ///
    /// # Args
    /// * `handles` - The handles for communicating with the nodes.
    async fn disconnect_nodes<'a, I, T>(&self, handles: I)
    where
        T: TransportLayer + 'a,
        I: Iterator<Item = &'a mut NodeHandle<T>> + 'a,
    {
        for node_handle in handles {
            let _ = node_handle.disconnect().await;
        }
    }
}
