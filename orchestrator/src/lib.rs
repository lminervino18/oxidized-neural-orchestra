pub mod configs;
pub mod dataset_format;
mod error;
mod session;

use configs::Adapter;
use dataset_format::{DatasetFormat, convert_to_binary};
use error::{OrchErr, Result};
pub use session::{Session, TrainedModel, TrainingEvent};

use crate::configs::{DatasetSrc, ModelConfig, TrainingConfig, Validator};

/// Starts the distributed training process and returns an active session.
///
/// If the dataset source is a local file with a known delimited format
/// (`.csv`, `.tsv`), it is transparently converted to a raw packed `f32`
/// binary file before validation and adaptation. The converted file is
/// placed next to the source with a `.bin` extension and reused on
/// subsequent runs.
///
/// # Args
/// * `model` - The model architecture configuration.
/// * `training` - The training configuration, including worker and server addresses.
///
/// # Returns
/// A new ongoing session.
///
/// # Errors
/// Returns an `OrchErr` if dataset conversion fails, config validation fails,
/// or connecting to any worker or server fails.
pub fn train(model: ModelConfig, mut training: TrainingConfig) -> Result<Session> {
    // Convert delimited datasets to binary before validation so the validator
    // always operates on raw packed f32 bytes.
    let converted_bin = if let DatasetSrc::Local {
        ref samples_path,
        ref labels_path,
    } = training.dataset.src
    {
        if let (Some(samples_format), Some(labels_format)) = (
            DatasetFormat::from_path(samples_path),
            DatasetFormat::from_path(labels_path),
        ) {
            let samples_bin_path =
                convert_to_binary(samples_path, samples_format).map_err(OrchErr::Io)?;
            let labels_bin_path =
                convert_to_binary(labels_path, labels_format).map_err(OrchErr::Io)?;
            training.dataset.src = DatasetSrc::Local {
                samples_path: samples_bin_path.clone(),
                labels_path: labels_bin_path.clone(),
            };
            (Some(samples_bin_path), Some(labels_bin_path))
        } else {
            (None, None)
        }
    } else {
        (None, None)
    };

    let validator = Validator::new();
    validator.validate(&model, &training)?;

    let input_size = training.dataset.x_size.get();

    let adapter = Adapter::new();
    let (workers, partitions, servers) = adapter.adapt_configs(model.clone(), &training)?;

    let session = Session::new(workers, partitions, servers, model, input_size)?;

    // All partitions have been sent to workers — the converted binary is no
    // longer needed and can be removed transparently.
    if let (Some(samples_bin_path), Some(labels_bin_path)) = converted_bin {
        if let Err(e) = std::fs::remove_file(&samples_bin_path) {
            log::warn!(
                "could not remove converted samples binary cache {}: {e}",
                samples_bin_path.display()
            );
        } else {
            log::info!(
                "removed converted samples binary cache {}",
                samples_bin_path.display()
            );
        }
        if let Err(e) = std::fs::remove_file(&labels_bin_path) {
            log::warn!(
                "could not remove converted labels binary cache {}: {e}",
                labels_bin_path.display()
            );
        } else {
            log::info!(
                "removed converted labels binary cache {}",
                labels_bin_path.display()
            );
        }
    }

    Ok(session)
}
