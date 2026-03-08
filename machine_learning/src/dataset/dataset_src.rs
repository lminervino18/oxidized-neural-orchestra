use super::inline_src::InlineSrc;
use rand::Rng;

/// The source of a dataset.
pub enum DatasetSrc {
    Inline(InlineSrc),
    // Stream(StreamSrc<R>),
}

impl DatasetSrc {
    /// Returns a new `DatasetSrc::Inline` dataset source.
    ///
    /// # Arguments
    /// * `data` - The buffer containing the dataset's raw data.
    pub fn inline(samples: Vec<f32>, labels: Vec<f32>) -> Self {
        DatasetSrc::Inline(InlineSrc::new(samples, labels))
    }
}

impl DatasetSrc {
    /// Returns the amount of values within the dataset source.
    pub fn len(&self) -> usize {
        match self {
            DatasetSrc::Inline(src) => src.len(),
            // DatasetSrc::Stream(_src) => todo!(),
        }
    }

    /// Shuffles the rows in the dataset using a random number generator.
    ///
    /// # Arguments
    /// * `rng` - A random number generator.
    pub fn shuffle<Rn: Rng>(&mut self, rows: usize, x_size: usize, y_size: usize, rng: &mut Rn) {
        match self {
            DatasetSrc::Inline(src) => src.shuffle(rows, x_size, y_size, rng),
            // DatasetSrc::Stream(_src) => todo!(),
        }
    }

    /// Retrieves a batch of the data source.
    ///
    /// # Arguments
    /// * `range` - The range of the data to be retrieved.
    ///
    /// # Returns
    /// A reference `&[f32]` to the raw data batch within the range.
    pub fn raw_batch(
        &self,
        row: usize,
        n: usize,
        x_size: usize,
        y_size: usize,
    ) -> (&[f32], &[f32]) {
        match self {
            DatasetSrc::Inline(src) => src.raw_batch(row, n, x_size, y_size),
            // DatasetSrc::Stream(_src) => todo!(),
        }
    }
}
