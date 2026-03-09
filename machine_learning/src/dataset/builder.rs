use super::{Dataset, DatasetSrc};
use comms::specs::machine_learning::DatasetSpec;
use std::num::NonZeroUsize;

#[derive(Default)]
pub struct DatasetBuilder;

impl DatasetBuilder {
    pub fn new() -> Self {
        Self
    }

    pub fn build_inline(&self, spec: DatasetSpec, dataset_raw: Vec<f32>) -> Dataset {
        let dataset_src = DatasetSrc::inline(dataset_raw);
        Dataset::new(
            dataset_src,
            NonZeroUsize::new(spec.x_size).unwrap(),
            NonZeroUsize::new(spec.y_size).unwrap(),
        )
    }
}
