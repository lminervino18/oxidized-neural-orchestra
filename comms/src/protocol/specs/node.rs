use std::{collections::HashMap, time::Duration};

use serde::{Deserialize, Serialize};

use super::{server::ServerSpec, worker::WorkerSpec};

/// A statistic request for a node to resolve.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StatRequest {
    Ping {
        addrs: Vec<String>,
        rounds: usize,
        incoming: usize,
    },
}

/// A statistic.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Stat<T> {
    pub min: T,
    pub max: T,
    pub mean: T,
}

/// A resolved statistic request from a node.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StatResponse {
    Ping {
        rtts: HashMap<String, Stat<Duration>>,
    },
}

/// The role assigned to an uninitialized node at bootstrap.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeSpec {
    Server(ServerSpec),
    Worker(WorkerSpec),
}
