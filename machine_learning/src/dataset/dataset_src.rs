use super::inmem_src::InMemSrc;
use rand::Rng;
use std::ops::Range;

/// The source of a dataset.
pub enum DatasetSrc {
    InMem(InMemSrc),
    // Stream(StreamSrc<R>),
}

impl DatasetSrc {
    /// Returns a new `DatasetSrc::Inline` dataset source.
    ///
    /// # Args
    /// * `data` - The buffer containing the dataset's raw data.
    pub fn inmem(samples: Vec<f32>, labels: Vec<f32>) -> Self {
        DatasetSrc::InMem(InMemSrc::new(samples, labels))
    }
}

impl DatasetSrc {
    /// Returns the amount of values in the dataset source.
    pub fn size(&self) -> usize {
        match self {
            DatasetSrc::InMem(src) => src.size(),
        }
    }

    /// Shuffles the rows in the dataset using a random number generator.
    ///
    /// # Args
    /// * `rng` - A random number generator.
    pub fn shuffle<Rn: Rng>(&mut self, rows: usize, x_size: usize, y_size: usize, rng: &mut Rn) {
        match self {
            DatasetSrc::InMem(src) => src.shuffle(rows, x_size, y_size, rng),
        }
    }

    /// Retrieves a batch of the data source.
    ///
    /// # Args
    /// * `range` - The range of the data to be retrieved.
    ///
    /// # Returns
    /// A reference `&[f32]` to the raw data batch within the range.
    pub fn raw_batch(&self, x_range: Range<usize>, y_range: Range<usize>) -> (&[f32], &[f32]) {
        match self {
            DatasetSrc::InMem(src) => src.raw_batch(x_range, y_range),
        }
    }
}
