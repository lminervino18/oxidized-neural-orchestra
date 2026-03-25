use std::num::NonZeroUsize;

use super::dataset_src::DatasetSrc;
use ndarray::{ArrayView2, Axis};
use rand::Rng;

/// A container for the *raw* dataset and its meta data. The raw data is expected to be structured
/// as rows, each with an x and it's expected output y.
pub struct Dataset {
    src: DatasetSrc,
    rows: usize,
    x_size: usize,
    row_size: usize,
}

impl Dataset {
    /// Creates a new `Dataset`.
    ///
    /// # Args
    /// * `data` - A vector containing the raw data.
    /// * `x_size` - Per row sample size.
    /// * `y_size` - Per row Label size.
    ///
    /// # Returns
    /// A new `Dataset` instance.
    pub fn new(src: DatasetSrc, x_size: NonZeroUsize, y_size: NonZeroUsize) -> Self {
        let row_size = x_size.saturating_add(y_size.get());

        Self {
            // SAFETY: row_size is a positive integer.
            rows: src.size() / row_size.get(),
            src,
            x_size: x_size.get(),
            row_size: row_size.get(),
        }
    }

    /// Shuffles the rows in the dataset using a random number generator.
    ///
    /// # Args
    /// * `rng` - A random number generator.
    pub fn shuffle<Rn: Rng>(&mut self, rng: &mut Rn) {
        self.src.shuffle(self.rows, self.row_size, rng);
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
            row_size,
            ref src,
            ..
        } = self;

        let offset = row * row_size;
        let raw_batch = src.raw_batch(offset..offset + row_size * n);

        let batch = ArrayView2::from_shape((n, row_size), raw_batch).unwrap();
        batch.split_at(Axis(1), x_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_inline_src_get_2rows() {
        let sums = [1.0, 2.0, 3.0, 3.0, 4.0, 7.0, 5.0, 6.0, 11.0];
        let x_size = NonZeroUsize::new(2).unwrap();
        let y_size = NonZeroUsize::new(1).unwrap();
        let src = DatasetSrc::inmem(sums.into());

        let ds = Dataset::new(src, x_size, y_size);

        let expected_x = ArrayView2::from_shape((2, 2), &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let expected_y = ArrayView2::from_shape((2, 1), &[3.0, 7.0]).unwrap();

        let (x, y) = ds.view_batch(0, 2);

        assert_eq!(x, expected_x);
        assert_eq!(y, expected_y);
    }
}
