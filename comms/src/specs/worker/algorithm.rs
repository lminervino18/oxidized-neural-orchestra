use std::net::SocketAddr;

use serde::{Deserialize, Serialize};

/// Distributed training algorithm selection.
///
/// Only `parameter_server` is currently implemented.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlgorithmSpec {
    ParameterServer { server_ips: Vec<SocketAddr> },
    AllReduce { peers: Vec<SocketAddr> },
    StrategySwitch,
}
