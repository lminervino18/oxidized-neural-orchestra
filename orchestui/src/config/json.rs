use std::fs;

use orchestrator::configs::{AlgorithmConfig, ModelConfig, TrainingConfig};

/// Parsed model architecture from `model.json`.
#[derive(Debug)]
pub struct ModelJson {
    pub config: ModelConfig,
}

/// Parsed training configuration from `training.json`.
#[derive(Debug)]
pub struct TrainingJson {
    pub config: TrainingConfig,
    pub worker_count: usize,
    pub server_count: usize,
}

/// Loads and parses a [`ModelConfig`] from a JSON file.
///
/// # Args
/// * `path` - Path to the model JSON file.
///
/// # Returns
/// A parsed `ModelJson` on success.
///
/// # Errors
/// Returns a human-readable string if the file cannot be read or parsed.
pub fn load_model(path: &str) -> Result<ModelJson, String> {
    let content = fs::read_to_string(path).map_err(|e| format!("cannot read '{path}': {e}"))?;
    let config: ModelConfig =
        serde_json::from_str(&content).map_err(|e| format!("invalid model config: {e}"))?;
    Ok(ModelJson { config })
}

/// Loads and parses a [`TrainingConfig`] from a JSON file.
///
/// # Args
/// * `path` - Path to the training JSON file.
///
/// # Returns
/// A parsed `TrainingJson` on success.
///
/// # Errors
/// Returns a human-readable string if the file cannot be read or parsed.
pub fn load_training(path: &str) -> Result<TrainingJson, String> {
    let content = fs::read_to_string(path).map_err(|e| format!("cannot read '{path}': {e}"))?;
    let config: TrainingConfig =
        serde_json::from_str(&content).map_err(|e| format!("invalid training config: {e}"))?;

    let (worker_count, server_count) = match &config.algorithm {
        AlgorithmConfig::ParameterServer { server_addrs, .. } => {
            (config.worker_addrs.len(), server_addrs.len())
        }
        AlgorithmConfig::StrategySwitch { server_addrs, .. } => {
            // All nodes start as AllReduce workers; server_addrs nodes are upgraded later.
            (config.worker_addrs.len() + server_addrs.len(), 0)
        }
        AlgorithmConfig::AllReduce => (config.worker_addrs.len(), 0),
    };

    Ok(TrainingJson {
        config,
        worker_count,
        server_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strategy_switch_worker_count_includes_server_addrs() {
        let json = r#"{
            "worker_addrs": ["127.0.0.1:50000", "127.0.0.1:50001"],
            "algorithm": {
                "strategy_switch": {
                    "server_addrs": ["127.0.0.1:40000"],
                    "synchronizer": "non_blocking",
                    "store": "wild"
                }
            },
            "serializer": "base",
            "dataset": {
                "src": { "inline": { "samples": [0.1, 0.2, 0.3, 0.4], "labels": [0.3] } },
                "x_size": 4,
                "y_size": 1
            },
            "optimizer": { "gradient_descent": { "lr": 0.01 } },
            "loss_fn": "mse",
            "batch_size": 4,
            "max_epochs": 10,
            "offline_epochs": 0,
            "seed": null,
            "early_stopping": null
        }"#;

        let config: TrainingConfig = serde_json::from_str(json).expect("parse failed");

        if let AlgorithmConfig::StrategySwitch { server_addrs, .. } = &config.algorithm {
            let worker_count = config.worker_addrs.len() + server_addrs.len();
            assert_eq!(worker_count, 3);
        } else {
            panic!("expected StrategySwitch");
        }
    }

    #[test]
    fn parameter_server_worker_count_excludes_server_addrs() {
        let json = r#"{
            "worker_addrs": ["127.0.0.1:50000"],
            "algorithm": {
                "parameter_server": {
                    "server_addrs": ["127.0.0.1:40000", "127.0.0.1:40001"],
                    "synchronizer": "barrier",
                    "store": "blocking"
                }
            },
            "serializer": "base",
            "dataset": {
                "src": { "inline": { "samples": [0.1, 0.2], "labels": [0.3] } },
                "x_size": 2,
                "y_size": 1
            },
            "optimizer": { "gradient_descent": { "lr": 0.01 } },
            "loss_fn": "mse",
            "batch_size": 4,
            "max_epochs": 10,
            "offline_epochs": 0,
            "seed": null,
            "early_stopping": null
        }"#;

        let config: TrainingConfig = serde_json::from_str(json).expect("parse failed");

        if let AlgorithmConfig::ParameterServer { server_addrs, .. } = &config.algorithm {
            let worker_count = config.worker_addrs.len();
            let server_count = server_addrs.len();
            assert_eq!(worker_count, 1);
            assert_eq!(server_count, 2);
        } else {
            panic!("expected ParameterServer");
        }
    }
}
