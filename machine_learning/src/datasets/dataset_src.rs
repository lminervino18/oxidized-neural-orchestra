use std::ops::Range;

use rand::Rng;

use super::inmem_src::InMemSrc;

/// The source of a dataset.
#[derive(Debug)]
pub enum DataSrc {
    InMem(InMemSrc),
    // Stream(StreamSrc<R>),
}

impl DataSrc {
    /// Returns a new `DataSrc::Inline` dataset source.
    ///
    /// # Args
    /// * `samples` - The buffer containing the dataset samples raw data.
    /// * `labels` - The buffer containing the dataset labels raw data.
    pub fn inmem(samples: Vec<f32>, labels: Vec<f32>) -> Self {
        DataSrc::InMem(InMemSrc::new(samples, labels))
    }

    /// Returns the amount of values in the dataset source.
    pub fn size(&self) -> usize {
        match self {
            DataSrc::InMem(src) => src.size(),
        }
    }

    /// Shuffles the rows in the dataset using a random number generator.
    ///
    /// # Args
    /// * `rows` - The amount of rows to shuffle.
    /// * `x_size` - The size of a sample.
    /// * `y_size` - The size of a label.
    /// * `rng` - A random number generator.
    pub fn shuffle<Rn: Rng>(&mut self, rows: usize, x_size: usize, y_size: usize, rng: &mut Rn) {
        match self {
            DataSrc::InMem(src) => src.shuffle(rows, x_size, y_size, rng),
        }
    }

    /// Retrieves a batch of the data source.
    ///
    /// # Args
    /// * `x_range` - The range of the samples to be retrieved.
    /// * `y_range` - The range of the labels to be retrieved.
    ///
    /// # Returns
    /// A reference `&[f32]` to the raw data batch within the range.
    pub fn raw_batch(&self, x_range: Range<usize>, y_range: Range<usize>) -> (&[f32], &[f32]) {
        match self {
            DataSrc::InMem(src) => src.raw_batch(x_range, y_range),
        }
    }

    /// Appends the given data source to self.
    ///
    /// # Args
    /// * `src` - The data to be appended.
    pub fn load(&mut self, src: Self) {
        match self {
            DataSrc::InMem(dst) => dst.load(src),
        }
    }
}
