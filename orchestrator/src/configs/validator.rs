use std::net::ToSocketAddrs;

use super::{AlgorithmConfig, DatasetConfig, ModelConfig, TrainingConfig};
use crate::{
    configs::training::DatasetSrc,
    error::{OrchErr, Result},
};

/// Validates orchestrator configs before adaptation, ensuring all invariants
/// are met before the training commences.
#[derive(Default)]
pub struct Validator;

impl Validator {
    /// Creates a new `Validator` instance.
    ///
    /// # Returns
    /// A new `Validator` instance.
    pub fn new() -> Self {
        Self
    }

    /// Validates model and training configs.
    ///
    /// # Args
    /// * `model` - The model architecture and initialization configuration.
    /// * `training` - The training configuration.
    ///
    /// # Errors
    /// An `OrchErr` if any invariant is unmet.
    pub fn validate<A: ToSocketAddrs>(
        &self,
        model: &ModelConfig,
        training: &TrainingConfig<A>,
    ) -> Result<()> {
        self.validate_model(model)?;
        self.validate_training(training)?;
        Ok(())
    }

    /// Validates the model's configuration.
    ///
    /// # Args
    /// * `model` - The model architecture and initialization configuration.
    ///
    /// # Errors
    /// An `OrchErr` if any invariant is unmet.
    fn validate_model(&self, model: &ModelConfig) -> Result<()> {
        if model.layers.is_empty() {
            return Err(OrchErr::InvalidConfig(
                "model must have at least one layer".into(),
            ));
        }

        Ok(())
    }

    /// Validates the training's configuration.
    ///
    /// # Errors
    /// An `OrchErr` if any training invariant is unmet.
    fn validate_training<A: ToSocketAddrs>(&self, training: &TrainingConfig<A>) -> Result<()> {
        if training.worker_addrs.is_empty() {
            let text = "at least one worker address is required".into();
            return Err(OrchErr::InvalidConfig(text));
        }

        let AlgorithmConfig::ParameterServer {
            ref server_addrs, ..
        } = training.algorithm;

        if server_addrs.is_empty() {
            let text = "at least one server address is required".into();
            return Err(OrchErr::InvalidConfig(text));
        }

        let DatasetConfig {
            ref src,
            x_size,
            y_size,
        } = training.dataset;

        let Some(row_size) = x_size.checked_add(y_size.get()) else {
            let text = "the row size for the dataset samples is larger than a usize".into();
            return Err(OrchErr::InvalidConfig(text));
        };

        let samples = match src {
            DatasetSrc::Inline { data } => {
                let len = data.len();

                if len % row_size != 0 {
                    let text = format!(
                        "dataset length ({len}) is not divisible by x_size + y_size ({row_size})"
                    );

                    return Err(OrchErr::InvalidConfig(text));
                }

                // SAFETY: row_size is a positive integer.
                len / row_size.get()
            }
            DatasetSrc::Local { path } => {
                let metadata = std::fs::metadata(path).map_err(|e| {
                    OrchErr::InvalidConfig(format!("cannot read dataset file: {e}"))
                })?;
                let len_bytes = metadata.len() as usize;
                let len = len_bytes / size_of::<f32>();

                if len % row_size != 0 {
                    let text = format!(
                        "dataset length ({len}) is not divisible by x_size + y_size ({row_size})"
                    );
                    return Err(OrchErr::InvalidConfig(text));
                }

                // SAFETY: row_size is a positive integer.
                len / row_size.get()
            }
        };

        if samples == 0 {
            let text = "dataset must have at least one sample".into();
            return Err(OrchErr::InvalidConfig(text));
        }

        let batch_size = training.batch_size.get();
        if batch_size > samples {
            let text =
                format!("batch_size ({batch_size}) exceeds dataset size ({samples} samples)");

            return Err(OrchErr::InvalidConfig(text));
        }

        Ok(())
    }
}
