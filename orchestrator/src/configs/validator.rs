use std::net::ToSocketAddrs;

use super::{
    AlgorithmConfig, DatasetConfig, LayerConfig, ModelConfig, SynchronizerConfig, TrainingConfig,
};
use crate::error::{OrchestratorError, Result};

/// Validates orchestrator configs before adaptation, ensuring all invariants
/// are met before any network connection is attempted.
pub struct Validator;

impl Validator {
    /// Creates a new `Validator` instance.
    pub fn new() -> Self {
        Self
    }

    /// Validates model and training configs.
    ///
    /// # Args
    /// * `model` - The model architecture configuration.
    /// * `training` - The training configuration.
    ///
    /// # Errors
    /// Returns an `OrchestratorError` if any invariant is violated.
    pub fn validate<A: ToSocketAddrs>(
        &self,
        model: &ModelConfig,
        training: &TrainingConfig<A>,
    ) -> Result<()> {
        self.validate_model(model)?;
        self.validate_training(training)
    }

    /// Validates the model configuration.
    ///
    /// # Errors
    /// Returns an `OrchestratorError` if the model has no layers or adjacent
    /// layers have incompatible dimensions.
    fn validate_model(&self, model: &ModelConfig) -> Result<()> {
        match model {
            ModelConfig::Sequential { layers } => {
                if layers.is_empty() {
                    return Err(OrchestratorError::InvalidConfig(
                        "model must have at least one layer".into(),
                    ));
                }

                // Adjacent layers must have compatible dimensions: prev.m == next.n
                for i in 1..layers.len() {
                    let prev_m = match layers[i - 1] {
                        LayerConfig::Dense { dim: (_, m), .. } => m,
                    };
                    let curr_n = match layers[i] {
                        LayerConfig::Dense { dim: (n, _), .. } => n,
                    };
                    if prev_m != curr_n {
                        return Err(OrchestratorError::InvalidConfig(format!(
                            "layer {i}: input size ({curr_n}) does not match \
                             previous layer output size ({prev_m})"
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    /// Validates the training configuration.
    ///
    /// # Errors
    /// Returns an `OrchestratorError` if worker or server address lists are empty,
    /// the barrier size is invalid, or the dataset is inconsistent with the batch size.
    fn validate_training<A: ToSocketAddrs>(&self, training: &TrainingConfig<A>) -> Result<()> {
        if training.worker_addrs.is_empty() {
            return Err(OrchestratorError::InvalidConfig(
                "at least one worker address is required".into(),
            ));
        }

        let AlgorithmConfig::ParameterServer {
            server_addrs,
            synchronizer,
            ..
        } = &training.algorithm;

        if server_addrs.is_empty() {
            return Err(OrchestratorError::InvalidConfig(
                "at least one server address is required".into(),
            ));
        }

        if let SynchronizerConfig::Barrier { barrier_size } = synchronizer {
            if *barrier_size > training.worker_addrs.len() {
                return Err(OrchestratorError::InvalidConfig(format!(
                    "barrier_size ({barrier_size}) cannot exceed number of workers ({})",
                    training.worker_addrs.len()
                )));
            }
            if *barrier_size == 0 {
                return Err(OrchestratorError::InvalidConfig(
                    "barrier_size must be greater than 0".into(),
                ));
            }
        }

        let dataset_samples = match &training.dataset {
            DatasetConfig::Inline {
                data,
                x_size,
                y_size,
            } => {
                let row_size = x_size + y_size;
                if row_size == 0 {
                    return Err(OrchestratorError::InvalidConfig(
                        "x_size + y_size must be greater than 0".into(),
                    ));
                }
                if data.len() % row_size != 0 {
                    return Err(OrchestratorError::InvalidConfig(format!(
                        "dataset length ({}) is not divisible by x_size + y_size ({row_size})",
                        data.len()
                    )));
                }
                data.len() / row_size
            }
            DatasetConfig::Local { path } => {
                return Err(OrchestratorError::InvalidConfig(format!(
                    "local dataset loading not yet implemented: {}",
                    path.display()
                )));
            }
        };

        if dataset_samples == 0 {
            return Err(OrchestratorError::InvalidConfig(
                "dataset must have at least one sample".into(),
            ));
        }

        if training.batch_size.get() > dataset_samples {
            return Err(OrchestratorError::InvalidConfig(format!(
                "batch_size ({}) exceeds dataset size ({dataset_samples} samples)",
                training.batch_size
            )));
        }

        Ok(())
    }
}
