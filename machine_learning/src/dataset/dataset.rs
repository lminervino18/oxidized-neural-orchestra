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
    /// # Arguments
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
            rows: src.len() / row_size.get(),
            src,
            x_size: x_size.get(),
            y_size: y_size.get(),
        }
    }

    /// Shuffles the rows in the dataset using a random number generator.
    ///
    /// # Arguments
    /// * `rng` - A random number generator.
    pub fn shuffle<Rn: Rng>(&mut self, rng: &mut Rn) {
        self.src.shuffle(self.rows, self.x_size, self.y_size, rng);
    }

    /// Retrieves the dataset in batches of size `batch_size`.
    ///
    /// # Arguments
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
    /// # Arguments
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

        let (x_raw, y_raw) = src.raw_batch(row, n, x_size, y_size);

        let x_batch = ArrayView2::from_shape((n, x_size), x_raw).unwrap();
        let y_batch = ArrayView2::from_shape((n, y_size), y_raw).unwrap();

        (x_batch, y_batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_inline_src_get_2rows() {
        let x_sums = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y_sums = [3.0, 7.0, 11.0];
        let x_size = NonZeroUsize::new(2).unwrap();
        let y_size = NonZeroUsize::new(1).unwrap();
        let src = DatasetSrc::inline(x_sums.into(), y_sums.into());

        let ds = Dataset::new(src, x_size, y_size);

        let expected_x = ArrayView2::from_shape((2, 2), &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let expected_y = ArrayView2::from_shape((2, 1), &[3.0, 7.0]).unwrap();

        let (x, y) = ds.view_batch(0, 2);

        assert_eq!(x, expected_x);
        assert_eq!(y, expected_y);
    }
}
