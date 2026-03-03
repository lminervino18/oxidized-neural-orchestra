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
    let content =
        std::fs::read_to_string(path).map_err(|e| format!("cannot read '{path}': {e}"))?;
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
    let content =
        std::fs::read_to_string(path).map_err(|e| format!("cannot read '{path}': {e}"))?;
    let config: TrainingConfig =
        serde_json::from_str(&content).map_err(|e| format!("invalid training config: {e}"))?;

    let (worker_count, server_count) = match &config.algorithm {
        AlgorithmConfig::ParameterServer { server_addrs, .. } => {
            (config.worker_addrs.len(), server_addrs.len())
        }
    };

    Ok(TrainingJson {
        config,
        worker_count,
        server_count,
    })
}
