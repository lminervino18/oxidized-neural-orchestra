use comms::specs::machine_learning::DatasetSpec;

use super::{Dataset, DatasetSrc};

#[derive(Default)]
pub struct DatasetBuilder;

impl DatasetBuilder {
    /// Creates a new `DatasetBuilder`.
    ///
    /// # Returns
    /// A new `DatasetBuilder` instance.
    pub fn new() -> Self {
        Self
    }

    /// Builds a `Dataset` from a vector with its raw data and a `DatasetSpec`.
    ///
    /// # Args
    /// * `spec` - The specification for the dataset.
    /// * `samples_raw` - The dataset samples raw data.
    /// * `labels_raw` - The dataset labels raw data.
    ///
    /// # Returns
    /// A fully initialized `Dataset` instance.
    pub fn build_inmem(
        &self,
        spec: DatasetSpec,
        samples_raw: Vec<f32>,
        labels_raw: Vec<f32>,
    ) -> Dataset {
        let dataset_src = DatasetSrc::inmem(samples_raw, labels_raw);
        Dataset::new(dataset_src, spec.x_size, spec.y_size)
    }
}
