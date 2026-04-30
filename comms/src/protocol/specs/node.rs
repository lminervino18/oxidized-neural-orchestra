use serde::{Deserialize, Serialize};

use super::{server::ServerSpec, worker::WorkerSpec};

/// The role assigned to an uninitialized node at bootstrap.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeSpec {
    Server(ServerSpec),
    Worker(WorkerSpec),
}
