use serde::{Deserialize, Serialize};

/// The specification for selecting and configuring a training strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategySpec {
    pub kind: String,
    #[serde(default)]
    pub params: serde_json::Value,
}
