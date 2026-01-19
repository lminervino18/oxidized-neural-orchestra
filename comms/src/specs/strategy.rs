use serde::{Deserialize, Serialize};

/// The specification for selecting and configuring a training strategy.
///
/// This is a wire-level contract used during worker bootstrap.
/// It intentionally avoids referencing any concrete strategy types so that
/// infrastructure crates do not depend on model/plugin crates.
///
/// `kind` is an opaque identifier (e.g. "noop", "mnist_mlp_v1") resolved by
/// a strategy factory on the worker side. `params` carries arbitrary JSON
/// configuration for that strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategySpec {
    pub kind: String,
    #[serde(default)]
    pub params: serde_json::Value,
}
