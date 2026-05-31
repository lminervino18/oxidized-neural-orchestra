use std::{
    collections::{BTreeMap, HashMap},
    time::Duration,
};

use comms::specs::node::StatResponse;

use super::{Graph, bipartite, tsp};

/// Calculates through node statistics the best role distribution and worker order.
pub struct Calculator {
    stats: BTreeMap<String, Vec<StatResponse>>,
}

/// The assignment return value for `Calculator::node_role_assignment` with
/// the network addresses of the nodes per role.
#[derive(Default)]
pub struct RoleAssignment {
    pub servers: Vec<String>,
    pub workers: Vec<String>,
}

impl Calculator {
    /// Creates a new `Calculator`.
    ///
    /// # Args
    /// * `stats` - The statistics for each address in the network.
    ///
    /// # Returns
    /// A new `Calculator` instance.
    pub fn new(stats: BTreeMap<String, Vec<StatResponse>>) -> Self {
        Self { stats }
    }

    /// Calculates the best role assignment for nodes in a Parameter Server training session.
    ///
    /// # Args
    /// * `servers` - The amout of servers in the current session.
    ///
    /// # Returns
    /// A `RoleAssignment` instance with the resultant node assignments.
    pub fn node_role_assignment(&self, servers: usize) -> RoleAssignment {
        let n = self.stats.len();
        let edges = self.node_delays();

        // SAFETY: All the edges were generated with valid indices.
        let graph = Graph::new(n, edges).unwrap();
        let server_addrs = bipartite::central_vertices(&graph, servers);
        let mut assignment = RoleAssignment::default();

        for (i, addr) in self.addrs().enumerate() {
            let buf = if server_addrs.contains(&i) {
                &mut assignment.servers
            } else {
                &mut assignment.workers
            };

            buf.push(addr.to_string());
        }

        assignment
    }

    /// Calculates the cheapest cycle for a ring of workers in an All Reduce training session.
    ///
    /// # Returns
    /// The ordered sequence of node addresses.
    pub fn worker_cycle(&self) -> Vec<String> {
        let n = self.stats.len();
        let edges = self.node_delays();

        // SAFETY: All the edges were generated with valid indices.
        let graph = Graph::new(n, edges).unwrap();
        let path = tsp::travelling_salesman_cycle(&graph);

        let addrs: Vec<_> = self.addrs().collect();
        path.into_iter().map(|i| addrs[i].to_string()).collect()
    }

    /// Calculates the edges given the addresses of the network.
    ///
    /// # Returns
    /// The node delays as edges `(v, w, weight)`.
    fn node_delays(&self) -> Vec<(usize, usize, Duration)> {
        let mut edges = Vec::new();

        let addr_to_index: HashMap<_, _> = self
            .addrs()
            .enumerate()
            .map(|(i, addr)| (addr, i))
            .collect();

        for (i, (_, stats)) in self.stats.iter().enumerate() {
            let pings = stats
                .iter()
                .flat_map(|stat| match stat {
                    StatResponse::Ping { rtts } => Some(rtts),
                    // NOTE: En caso de agregar más variantes, acá
                    //       deberían devolver `None`. Descomentar
                    //       la siguiente linea.
                    //
                    // _ => None,
                })
                .flatten();

            for (addr, dur_stat) in pings {
                let j = addr_to_index[addr.as_str()];
                edges.push((i, j, dur_stat.max));
            }
        }

        edges
    }

    /// Yields an iterator of the node addresses.
    ///
    /// # Returns
    /// An iterator over all the node addresses.
    fn addrs(&self) -> impl Iterator<Item = &str> {
        self.stats.keys().map(|addr| addr.as_str())
    }
}
