use std::{collections::HashMap, io, time::Duration};

use comms::{
    Acceptor, Connection, Connector, NodeEvent, NodeHandle, TransportLayer,
    protocol::Entity,
    specs::node::{Stat, StatRequest, StatResponse},
};
use futures::future;
use log::{debug, warn};
use tokio::{
    io::{AsyncRead, AsyncWrite},
    net::tcp::{OwnedReadHalf, OwnedWriteHalf},
    time::Instant,
};

type R = OwnedReadHalf;
type W = OwnedWriteHalf;

/// Resolves the requested statistic calculations by the orchestrator.
pub struct StatService<'a, T, F, G, Fut>
where
    T: TransportLayer<R, W>,
    F: Fn(R, W) -> T + Clone,
    G: Fn() -> Fut + Clone,
    Fut: Future<Output = io::Result<(R, W)>>,
{
    acceptor: &'a mut Acceptor<T, F, G, Fut>,
    connector: &'a mut Connector<T, F>,
}

/// A struct helper to mantain the address of a handle close to it.
struct AddressedHandle<T>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
    T: TransportLayer<R, W>,
{
    addr: String,
    handle: NodeHandle<R, W, T>,
}

impl<'a, T, F, G, Fut> StatService<'a, T, F, G, Fut>
where
    T: TransportLayer<R, W>,
    F: Fn(R, W) -> T + Clone,
    G: Fn() -> Fut + Clone,
    Fut: Future<Output = io::Result<(R, W)>>,
{
    /// Creates a new `StatServicer`.
    ///
    /// # Args
    /// * `acceptor` - The acceptor used to receive incoming node connections.
    /// * `connector` - The connector used to reach other nodes.
    ///
    /// # Returns
    /// A new `StatServicer` instance.
    pub fn new(
        acceptor: &'a mut Acceptor<T, F, G, Fut>,
        connector: &'a mut Connector<T, F>,
    ) -> Self {
        Self {
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
            if let Ok(stat) = self.serve_one(req).await {
                stats.push(stat);
            }
        }

        stats
    }

    /// Serves a single request.
    ///
    /// # Args
    /// * `req` - The request to serve.
    ///
    /// # Returns
    /// The response for the request or an io error if occurred.
    async fn serve_one(&mut self, req: StatRequest) -> io::Result<StatResponse> {
        match req {
            StatRequest::Ping {
                addrs,
                rounds,
                incoming,
            } => Ok(self.serve_ping(addrs, rounds, incoming).await),
        }
    }

    /// Resolves the ping request by sending `rounds` pings to the node at `addr`.
    ///
    /// # Args
    /// * `addrs` - The network addresses of the other nodes to ping.
    /// * `rounds` - The amount of pings to make per node.
    /// * `incoming` - The amount of pings to listen to.
    ///
    /// # Returns
    /// The durations for each of the pings made in a `StatResponse`.
    async fn serve_ping(
        &mut self,
        addrs: Vec<String>,
        rounds: usize,
        incoming: usize,
    ) -> StatResponse {
        debug!("Serving ping");
        let (mut ping_handles, mut pong_handles) = self.establish_ping_conns(addrs, incoming).await;
        debug!("done with connections");
        let mut rtts = HashMap::new();

        for i in 0..rounds {
            debug!("Serving ping round {i}/{rounds}");
            self.serve_ping_round(&mut ping_handles, &mut pong_handles, &mut rtts)
                .await;
        }

        let rtts = rtts
            .into_iter()
            .flat_map(|(addr, samples)| {
                if samples.is_empty() {
                    return None;
                }

                let mut min = samples[0];
                let mut max = samples[0];
                let mut sum = samples[0];

                for &dur in samples.iter() {
                    min = min.min(dur);
                    max = max.max(dur);
                    sum += dur;
                }

                let stat = Stat {
                    min,
                    max,
                    mean: sum.checked_div(samples.len() as u32)?,
                };

                Some((addr, stat))
            })
            .collect();

        StatResponse::Ping { rtts }
    }

    /// Establishes the incoming and outgoing connections with the rest of the nodes.
    ///
    /// # Args
    /// * `addrs` - The network addresses of the other nodes to ping.
    /// * `incoming` - The amount of pings to listen to.
    ///
    /// # Returns
    /// Both the ping and pong handles for the ping calculations.
    async fn establish_ping_conns(
        &mut self,
        addrs: Vec<String>,
        incoming: usize,
    ) -> (Vec<AddressedHandle<T>>, Vec<NodeHandle<R, W, T>>) {
        let mut ping_handles = Vec::with_capacity(addrs.len());

        for addr in addrs {
            match self.connector.connect_node(&addr, Entity::Node).await {
                Ok(handle) => {
                    let addr_handle = AddressedHandle { addr, handle };
                    ping_handles.push(addr_handle);
                }
                Err(e) => warn!("failed to connect to node at {addr}: {e}"),
            }
        }

        let mut pong_handles = Vec::with_capacity(incoming);
        let src = Entity::Node;

        for _ in 0..incoming {
            if let Ok(Connection::Node(node_handle)) = self.acceptor.accept(src).await {
                pong_handles.push(node_handle);
            }
        }

        (ping_handles, pong_handles)
    }

    /// Runs a pinging round between the nodes.
    ///
    /// # Args
    /// * `ping_handles` - The handles to which to ping.
    /// * `pong_handles` - The handles to which to listen for pings and return pongs.
    /// * `rtts` - The round trip time storage.
    async fn serve_ping_round(
        &mut self,
        ping_handles: &mut Vec<AddressedHandle<T>>,
        pong_handles: &mut Vec<NodeHandle<R, W, T>>,
        rtts: &mut HashMap<String, Vec<Duration>>,
    ) {
        let ping_futs = ping_handles
            .iter_mut()
            .map(async |addr_handle| self.ping_node(&mut addr_handle.handle).await);

        let pong_futs = pong_handles
            .iter_mut()
            .map(async |node_handle| self.pong_node(node_handle).await);

        let (pings, pongs) = tokio::join!(future::join_all(ping_futs), future::join_all(pong_futs));
        let mut removed = 0;

        for (i, ping) in pings.into_iter().enumerate() {
            let AddressedHandle { addr, .. } = &ping_handles[i - removed];

            match ping {
                Ok(dur) => match rtts.get_mut(addr) {
                    Some(durs) => durs.push(dur),
                    None => {
                        rtts.insert(addr.clone(), vec![dur]);
                    }
                },
                Err(e) => {
                    warn!("ping failed for node {addr}: {e}");
                    ping_handles.remove(i - removed);
                    removed += 1;
                }
            }
        }

        let mut removed = 0;

        for (i, pong) in pongs.into_iter().enumerate() {
            if let Err(e) = pong {
                warn!("pong failed: {e}");
                pong_handles.remove(i - removed);
                removed += 1;
            }
        }
    }

    /// Pings a node given it's handle and waits for a pong response.
    ///
    /// # Args
    /// * `node_handle` - The node's handle to ping and listen.
    ///
    /// # Returns
    /// The round trip time duration or an io error if occurred.
    async fn ping_node(&self, node_handle: &mut NodeHandle<R, W, T>) -> io::Result<Duration> {
        let start = Instant::now();
        node_handle.ping().await?;

        match node_handle.recv_event().await? {
            NodeEvent::Pong => Ok(start.elapsed()),
            event => {
                let text = format!("unexpected event from peer node: {event:?}");
                Err(io::Error::other(text))
            }
        }
    }

    /// Waits or a ping event from the given node's handle and returns a pong.
    ///
    /// # Args
    /// * `node_handle` - The onde's handle to listen and pong.
    ///
    /// # Returns
    /// An io error if occurred.
    async fn pong_node(&self, node_handle: &mut NodeHandle<R, W, T>) -> io::Result<()> {
        match node_handle.recv_event().await? {
            NodeEvent::Ping => node_handle.pong().await,
            event => {
                let text = format!("unexpected event from peer node: {event:?}");
                Err(io::Error::other(text))
            }
        }
    }
}
