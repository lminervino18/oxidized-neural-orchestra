use std::time::Duration;

use serde::{Deserialize, Serialize};

use super::{server::ServerSpec, worker::WorkerSpec};

/// A statistic request for a node to resolve.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StatRequest {
    Ping { addr: String, times: usize },
}

/// A resolved statistic request from a node.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StatResponse {
    Pong { addr: String, rtts: Vec<Duration> },
}

/// The role assigned to an uninitialized node at bootstrap.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeSpec {
    Server(ServerSpec),
    Worker(WorkerSpec),
}
