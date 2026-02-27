use std::num::NonZeroUsize;

use ndarray::{ArrayView2, Axis};
use rand::Rng;

/// A container for the *raw* dataset and its meta data. The raw data is expected to be structured
/// as rows, each with an x and it's expected output y.
pub struct Dataset {
    data: Vec<f32>,
    rows: usize,
    x_size: usize,
    row_size: usize,
}

impl Dataset {
    /// Creates a new `Dataset`.
    ///
    /// # Arguments
    /// * `data` - A vector containing the raw data.
    /// * `x_size` - Per row sample size.
    /// * `y_size` - Per row Label size.
    pub fn new(data: Vec<f32>, x_size: usize, y_size: usize) -> Self {
        let row_size = x_size + y_size;

        Self {
            rows: data.len() / row_size,
            data,
            x_size,
            row_size,
        }
    }

    /// Shuffles the rows in the dataset using a random number generator.
    ///
    /// # Arguments
    /// * `rng` - A random number generator.
    pub fn shuffle<R: Rng>(&mut self, rng: &mut R) {
        let Self {
            rows,
            row_size,
            ref mut data,
            ..
        } = *self;

        for i in 0..rows {
            let j = rng.random_range(i..rows);
            if i == j {
                continue;
            }

            let (i, j) = (i.min(j), i.max(j));
            let i_data = i * row_size;
            let j_data = j * row_size;

            let (left, right) = data.split_at_mut(j_data);
            let row = &mut left[i_data..i_data + row_size];
            let other = &mut right[..row_size];
            row.swap_with_slice(other);
        }
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
            row_size,
            ref data,
            ..
        } = self;

        let offset = row * row_size;
        let raw_batch = &data[offset..offset + row_size * n];

        let batch = ArrayView2::from_shape((n, row_size), raw_batch).unwrap();
        batch.split_at(Axis(1), x_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_get_2rows() {
        let sums = [1.0, 2.0, 3.0, 3.0, 4.0, 7.0, 5.0, 6.0, 11.0];

        let ds = Dataset::new(sums.into(), 2, 1);

        let expected_x = ArrayView2::from_shape((2, 2), &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let expected_y = ArrayView2::from_shape((2, 1), &[3.0, 7.0]).unwrap();

        let (x, y) = ds.view_batch(0, 2);

        assert_eq!(x, expected_x);
        assert_eq!(y, expected_y);
    }
}
