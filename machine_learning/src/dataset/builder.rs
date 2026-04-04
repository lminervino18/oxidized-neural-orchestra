use super::{Dataset, DatasetSrc};
use comms::specs::machine_learning::DatasetSpec;

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
    /// * `dataset_raw` - The dataset's raw data.
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
