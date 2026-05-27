use std::{collections::HashMap, io, time::Duration};

use comms::{
    Acceptor, Connector, NodeEvent, NodeHandle, TransportLayer,
    protocol::Entity,
    specs::node::{StatRequest, StatResponse},
};
use futures::future;
use log::warn;
use tokio::{
    net::{
        TcpStream,
        tcp::{OwnedReadHalf, OwnedWriteHalf},
    },
    time::Instant,
};

type R = OwnedReadHalf;
type W = OwnedWriteHalf;

/// Resolves the requested statistic calculations by the orchestrator.
pub struct StatService<'a, T, F, G>
where
    T: TransportLayer,
    F: AsyncFn() -> io::Result<T>,
    G: Fn(R, W) -> T,
{
    id: usize,
    acceptor: &'a mut Acceptor<T, F>,
    connector: &'a mut Connector<R, W, T, G>,
}

impl<'a, T, F, G> StatService<'a, T, F, G>
where
    T: TransportLayer,
    F: AsyncFn() -> io::Result<T>,
    G: Fn(R, W) -> T,
{
    /// Creates a new `StatServicer`.
    ///
    /// # Args
    /// * `id` - This node's id.
    /// * `acceptor` - The acceptor used to receive incoming node connections.
    /// * `connector` - The connector used to reach other nodes.
    ///
    /// # Returns
    /// A new `StatServicer` instance.
    pub fn new(
        id: usize,
        acceptor: &'a mut Acceptor<T, F>,
        connector: &'a mut Connector<R, W, T, G>,
    ) -> Self {
        Self {
            id,
            acceptor,
            connector,
        }
    }

    /// Serves the given `StatRequest`s and generates a `StatResponse` for each of them.
    ///
    /// # Args
    /// * `reqs` - The requests to resolve.
    ///
    /// # Returns
    /// The resolved requests in the form of responses or an io error if occurred.
    pub async fn serve(&mut self, reqs: Vec<StatRequest>) -> Vec<StatResponse> {
        let mut stats = Vec::with_capacity(reqs.len());

        for req in reqs {
            if let Ok(stat) = self.resolve(req).await {
                stats.push(stat);
            }
        }

        stats
    }

    /// Resolves a single request.
    ///
    /// # Args
    /// * `req` - The request to resolve.
    ///
    /// # Returns
    /// The response for the request or an io error if occurred.
    async fn resolve(&mut self, req: StatRequest) -> io::Result<StatResponse> {
        match req {
            StatRequest::Ping { addrs, times } => self.resolve_pings(addrs, times).await,
        }
    }

    /// Resolves the ping request by sending `times` pings to the node at `addr`.
    ///
    /// # Args
    /// * `id` - The other node's id.
    /// * `addrs` - The network addresses of the other nodes to ping.
    /// * `times` - The amount of pings to make.
    ///
    /// # Returns
    /// The durations for each of the pings made in a `StatResponse`.
    async fn resolve_pings(
        &mut self,
        addrs: Vec<String>,
        times: usize,
    ) -> io::Result<StatResponse> {
        let mut handles = Vec::with_capacity(addrs.len());

        for (id, addr) in addrs.iter().enumerate() {
            match self.connect_node(id, addr).await {
                Ok(node_handle) => handles.push(node_handle),
                Err(e) => warn!("failed to connect to node at {addr}: {e}"),
            }
        }

        let pingers = handles.into_iter().map(async |mut node_handle| {
            let start = Instant::now();
            node_handle.ping().await?;

            match node_handle.recv_event().await? {
                NodeEvent::Pong => {}
            }

            Ok::<_, io::Error>(start.elapsed())
        });

        // TODO: Hay que ver como hacer para escuchar por pings ajenos y
        //       ser capaz de enviar y medir cuanto tarda la respuesta.

        todo!()
    }

    /// Connects to another node.
    ///
    /// # Args
    /// * `id` - The other node's id.
    /// * `addr` - The network address of th node to connect to.
    ///
    /// # Returns
    /// A handle to the other node or an io error if occurred.
    async fn connect_node(&self, id: usize, addr: &str) -> io::Result<NodeHandle<T>> {
        let stream = TcpStream::connect(addr).await?;
        let (rx, tx) = stream.into_split();

        let node_handle = self
            .connector
            .connect_node(id, rx, tx, Entity::Node { id: self.id })
            .await?;

        Ok(node_handle)
    }
}
