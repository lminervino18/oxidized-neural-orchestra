use serde::{Deserialize, Serialize};

/// The specification for the parameter server distributed training algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterServerSpec {
    pub server_addrs: Vec<String>,
    pub server_sizes: Vec<usize>,
    pub server_ordering: Vec<usize>,
}

/// The specification for the ring all-reduce distributed training algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RingAllReduceSpec {
    pub worker_addrs: Vec<String>,
    pub ring_ordering: Vec<usize>,
}

/// Distributed training algorithm selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlgorithmSpec {
    ParameterServer(ParameterServerSpec),
    RingAllReduce(RingAllReduceSpec),
}
