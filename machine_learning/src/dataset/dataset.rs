use std::num::NonZeroUsize;

use ndarray::ArrayView2;
use rand::Rng;

use super::{dataset_src::DataSrc, inmem_src::InMemSrc};

/// A container for the *raw* dataset and its meta data. The raw data is expected to be structured
/// as rows, each with an x and it's expected output y.
pub struct Dataset {
    src: DataSrc,
    rows: usize,
    x_size: NonZeroUsize,
    y_size: NonZeroUsize,
}

impl Dataset {
    /// Creates a new `Dataset`.
    ///
    /// # Args
    /// * `x_size` - Per row sample size.
    /// * `y_size` - Per row Label size.
    ///
    /// # Returns
    /// A new `Dataset` instance.
    pub fn new(x_size: NonZeroUsize, y_size: NonZeroUsize) -> Self {
        Self {
            rows: 0,
            src: DataSrc::InMem(InMemSrc::default()),
            x_size,
            y_size,
        }
    }

    /// Creates a new `Dataset`.
    ///
    /// # Args
    /// * `src` - The dataset's source data.
    /// * `x_size` - Per row sample size.
    /// * `y_size` - Per row Label size.
    ///
    /// # Returns
    /// A new `Dataset` instance.
    pub fn loaded(src: DataSrc, x_size: NonZeroUsize, y_size: NonZeroUsize) -> Self {
        Self {
            // SAFETY: The divisor is always greater than 0.
            rows: src.size() / (x_size.get() + y_size.get()),
            src,
            x_size,
            y_size,
        }
    }

    /// Appends this data source to this dataset.
    ///
    /// # Args
    /// * `src` - The data to append.
    pub fn load(&mut self, src: DataSrc) {
        let (x_size, y_size) = self.sizes();
        self.rows += src.size() / (x_size.get() + y_size.get());
        self.src.load(src);
    }

    /// Shuffles the rows in the dataset using a random number generator.
    ///
    /// # Args
    /// * `rng` - A random number generator.
    pub fn shuffle<R: Rng>(&mut self, rng: &mut R) {
        let (x_size, y_size) = self.sizes();
        self.src.shuffle(self.rows, x_size.get(), y_size.get(), rng);
    }

    /// Retrieves the dataset in batches of size `batch_size`.
    ///
    /// # Args
    /// * `batch_size` - The maximum size of batches to yield.
    ///
    /// # Returns
    /// An iterator over the batches of the dataset in the form of tuples of `ArrayView2`s.
    pub fn batches<'a>(
        &'a self,
        batch_size: NonZeroUsize,
    ) -> impl Iterator<Item = (ArrayView2<'a, f32>, ArrayView2<'a, f32>)> + 'a {
        (0..self.rows).step_by(batch_size.get()).map(move |start| {
            let n = (start + batch_size.get()).min(self.rows) - start;
            self.view_batch(start, n)
        })
    }

    /// Partitions the dataset into n parts minimizing the size between them all.
    ///
    /// # Args
    /// * `n` - The amount of partitions to make.
    ///
    /// # Returns
    /// An iterator over the dataset's raw partitions.
    pub fn partition(&self, n: usize) -> impl Iterator<Item = (&[f32], &[f32])> {
        let (x_size, y_size) = self.sizes();

        let batch_size = self.rows / n.max(1);
        let extra = self.rows % n.max(1);
        let mut start = 0;

        (0..n).map(move |i| {
            let end = start + batch_size + (i < extra) as usize;
            let samples_range = start * x_size.get()..end * x_size.get();
            let labels_range = start * y_size.get()..end * x_size.get();

            start = end;
            self.src.raw_batch(samples_range, labels_range)
        })
    }

    pub fn sizes(&self) -> (NonZeroUsize, NonZeroUsize) {
        (self.x_size, self.y_size)
    }

    /// Creates a view over a certain batch of the dataset.
    ///
    /// # Args
    /// * `row` - The starting row to retrieve.
    /// * `n` - The amount of rows to keep in the view.
    ///
    /// # Returns
    /// A tuple of both samples and labels inside the selected batch.
    fn view_batch<'a>(
        &'a self,
        row: usize,
        n: usize,
    ) -> (ArrayView2<'a, f32>, ArrayView2<'a, f32>) {
        let &Self {
            x_size,
            y_size,
            ref src,
            ..
        } = self;

        let x_offset = row * x_size.get();
        let y_offset = row * y_size.get();
        let x_range = x_offset..x_offset + x_size.get() * n;
        let y_range = y_offset..y_offset + y_size.get() * n;
        let (x_raw_batch, y_raw_batch) = src.raw_batch(x_range, y_range);

        let x_batch = ArrayView2::from_shape((n, x_size.get()), x_raw_batch).unwrap();
        let y_batch = ArrayView2::from_shape((n, y_size.get()), y_raw_batch).unwrap();

        (x_batch, y_batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::aview2;

    #[test]
    fn test_dataset_inline_src_get_2rows() {
        let x_sums = vec![1., 2., 3., 4., 5., 6.];
        let y_sums = vec![3., 7., 11.];
        let x_size = NonZeroUsize::new(2).unwrap();
        let y_size = NonZeroUsize::new(1).unwrap();
        let src = DataSrc::inmem(x_sums, y_sums);

        let ds = Dataset::loaded(src, x_size, y_size);

        let expected_x = aview2(&[[1., 2.], [3., 4.]]);
        let expected_y = aview2(&[[3.], [7.]]);

        let (x, y) = ds.view_batch(0, 2);

        assert_eq!(x, expected_x);
        assert_eq!(y, expected_y);
    }
}
