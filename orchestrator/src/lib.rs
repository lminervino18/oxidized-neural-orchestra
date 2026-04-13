pub mod configs;
use std::path::PathBuf;
pub mod dataset_format;
mod error;
mod session;

use std::fs;

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
    let dataset_bin = generate_binary_dataset(&mut training.dataset.src);
    let validator = Validator::new();
    validator.validate(&model, &training)?;

    let input_size = training.dataset.x_size.get();

    let adapter = Adapter::new();
    let (workers, partitions, servers) = adapter.adapt_configs(model.clone(), &training)?;

    let session = Session::new(
        workers,
        partitions,
        servers,
        model,
        input_size,
        training.algorithm.clone(),
    )?;

    if let Some((samples_bin, labels_bin)) = dataset_bin {
        remove_binary(&samples_bin);
        remove_binary(&labels_bin);
    }

    Ok(session)
}

/// Converts delimited dataset samples and labels so the validator always operates on raw packed
/// f32 bytes.
///
/// # Args
/// * `src` - The dataset source.
///
/// # Returns
/// The paths of the generated binary files if they were found or `None` if not.
fn generate_binary_dataset(src: &mut DatasetSrc) -> Option<(PathBuf, PathBuf)> {
    let DatasetSrc::Local {
        samples_path,
        labels_path,
    } = src
    else {
        return None;
    };

    let (Some(samples_format), Some(labels_format)) = (
        DatasetFormat::from_path(samples_path),
        DatasetFormat::from_path(labels_path),
    ) else {
        return None;
    };

    let samples_bin_path = convert_to_binary(samples_path, samples_format).ok()?;
    let labels_bin_path = convert_to_binary(labels_path, labels_format).ok()?;
    *src = DatasetSrc::Local {
        samples_path: samples_bin_path.clone(),
        labels_path: labels_bin_path.clone(),
    };

    Some((samples_bin_path, labels_bin_path))
}

/// Removes a binary dataset file.
///
/// # Args
/// * `path` - The path of the file to remove.
fn remove_binary(path: &PathBuf) {
    if let Err(e) = fs::remove_file(path) {
        log::warn!(
            "could not remove converted dataset binary cache {}: {e}",
            path.display()
        );
    } else {
        log::info!("removed converted dataset binary cache {}", path.display());
    }
}
