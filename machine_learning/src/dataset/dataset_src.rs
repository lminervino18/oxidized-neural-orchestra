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
    /// # Arguments
    /// * `data` - The buffer containing the dataset's raw data.
    pub fn inmem(data: Vec<f32>) -> Self {
        DatasetSrc::InMem(InMemSrc::new(data))
    }
}

impl DatasetSrc {
    /// Returns the amount of values within the dataset source.
    pub fn size(&self) -> usize {
        match self {
            DatasetSrc::InMem(src) => src.len(),
        }
    }

    /// Shuffles the rows in the dataset using a random number generator.
    ///
    /// # Arguments
    /// * `rng` - A random number generator.
    pub fn shuffle<Rn: Rng>(&mut self, rows: usize, row_size: usize, rng: &mut Rn) {
        match self {
            DatasetSrc::InMem(src) => src.shuffle(rows, row_size, rng),
        }
    }

    /// Retrieves a batch of the data source.
    ///
    /// # Arguments
    /// * `range` - The range of the data to be retrieved.
    ///
    /// # Returns
    /// A reference `&[f32]` to the raw data batch within the range.
    pub fn raw_batch(&self, range: Range<usize>) -> &[f32] {
        match self {
            DatasetSrc::InMem(src) => src.raw_batch(range),
        }
    }
}
