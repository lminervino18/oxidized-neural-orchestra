use std::fs;

use super::{AlgorithmConfig, DataSrc, DatasetConfig, LayerConfig, ModelConfig, TrainingConfig};
use crate::error::{OrchErr, Result};

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
    pub fn validate(&self, model: &ModelConfig, training: &TrainingConfig) -> Result<()> {
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

        for layer in &model.layers {
            match layer {
                LayerConfig::Dense { .. } => {
                    continue;
                }
                LayerConfig::Conv {
                    input_dim: (_, height, width),
                    kernel_dim: (_, _, size),
                    padding,
                    ..
                } => {
                    let remaining_height = (height.get() + 2 * padding).saturating_sub(size.get());
                    let remaining_width = (width.get() + 2 * padding).saturating_sub(size.get());

                    if remaining_height == 0 || remaining_width == 0 {
                        let details = "conv layer input_dim + 2 * padding - kernel_size must be greater than 0 for both height and width";
                        return Err(OrchErr::InvalidConfig(details.into()));
                    }
                }
            }
        }

        Ok(())
    }

    /// Validates the training's configuration.
    ///
    /// * `training` - The training configuration.
    ///
    /// # Errors
    /// An `OrchErr` if any training invariant is unmet.
    fn validate_training(&self, training: &TrainingConfig) -> Result<()> {
        if training.addrs.is_empty() {
            let text = "at least one network address is required".into();
            return Err(OrchErr::InvalidConfig(text));
        }

        if let AlgorithmConfig::ParameterServer { nservers, .. }
        | AlgorithmConfig::StrategySwitch { nservers, .. } = training.algorithm
            && training.addrs.len().saturating_sub(nservers.get()) == 0
        {
            let text = "the amount of workers must be a positive number".into();
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
            DataSrc::Inline { samples, labels } => {
                let len = samples.len() + labels.len();

                if len % row_size != 0 {
                    let text = format!(
                        "dataset length ({len}) is not divisible by x_size + y_size ({row_size})"
                    );

                    return Err(OrchErr::InvalidConfig(text));
                }

                // SAFETY: row_size is a positive integer.
                len / row_size.get()
            }
            DataSrc::Local {
                samples_path,
                labels_path,
            } => {
                let samples_metadata = fs::metadata(samples_path).map_err(|e| {
                    OrchErr::InvalidConfig(format!("cannot read dataset samples file: {e}"))
                })?;
                let labels_metadata = fs::metadata(labels_path).map_err(|e| {
                    OrchErr::InvalidConfig(format!("cannot read dataset labels file: {e}"))
                })?;
                let samples_len_bytes = samples_metadata.len() as usize;
                let labels_len_bytes = labels_metadata.len() as usize;
                let len = (samples_len_bytes + labels_len_bytes) / size_of::<f32>();

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
