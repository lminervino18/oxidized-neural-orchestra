use std::num::NonZeroUsize;

use super::dataset_src::DatasetSrc;
use ndarray::ArrayView2;
use rand::Rng;

/// A container for the *raw* dataset and its meta data. The raw data is expected to be structured
/// as rows, each with an x and it's expected output y.
pub struct Dataset {
    src: DatasetSrc,
    rows: usize,
    x_size: usize,
    y_size: usize,
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
        let x_size = x_size.get();
        let y_size = y_size.get();
        // SAFETY: both x_size and y_size are positive integers.
        let rows = src.size() / (x_size + y_size);

        Self {
            rows,
            src,
            x_size,
            y_size,
        }
    }

    /// Shuffles the rows in the dataset using a random number generator.
    ///
    /// # Args
    /// * `rng` - A random number generator.
    pub fn shuffle<Rn: Rng>(&mut self, rng: &mut Rn) {
        self.src.shuffle(self.rows, self.x_size, self.y_size, rng);
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
            y_size,
            ref src,
            ..
        } = self;

        let x_offset = row * x_size;
        let y_offset = row * y_size;
        let x_range = x_offset..x_offset + x_size * n;
        let y_range = y_offset..y_offset + y_size * n;
        let (x_raw_batch, y_raw_batch) = src.raw_batch(x_range, y_range);

        let x_batch = ArrayView2::from_shape((n, x_size), x_raw_batch).unwrap();
        let y_batch = ArrayView2::from_shape((n, y_size), y_raw_batch).unwrap();

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
        let src = DatasetSrc::inmem(x_sums, y_sums);

        let ds = Dataset::new(src, x_size, y_size);

        let expected_x = aview2(&[[1., 2.], [3., 4.]]);
        let expected_y = aview2(&[[3.], [7.]]);

        let (x, y) = ds.view_batch(0, 2);

        assert_eq!(x, expected_x);
        assert_eq!(y, expected_y);
    }
}
