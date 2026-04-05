use std::ops::Range;

use rand::Rng;

pub struct InMemSrc {
    samples: Vec<f32>,
    labels: Vec<f32>,
}

impl InMemSrc {
    pub fn new(samples: Vec<f32>, labels: Vec<f32>) -> Self {
        InMemSrc { samples, labels }
    }

    /// Returns the sum of the length of the raw samples and labels.
    pub fn size(&self) -> usize {
        self.samples.len() + self.labels.len()
    }

    pub fn shuffle<R: Rng>(&mut self, rows: usize, x_size: usize, y_size: usize, rng: &mut R) {
        for i in 0..rows {
            let j = rng.random_range(i..rows);
            if i == j {
                continue;
            }

            let (i, j) = (i.min(j), i.max(j));

            Self::shuffle_slice(&mut self.samples, i, j, x_size);
            Self::shuffle_slice(&mut self.labels, i, j, y_size);
        }
    }

    fn shuffle_slice(slice: &mut [f32], min: usize, max: usize, size: usize) {
        let i_data = min * size;
        let j_data = max * size;

        let (left, right) = slice.split_at_mut(j_data);

        let row = &mut left[i_data..i_data + size];
        let other = &mut right[..size];

        row.swap_with_slice(other);
    }

    pub fn raw_batch(&self, x_range: Range<usize>, y_range: Range<usize>) -> (&[f32], &[f32]) {
        (&self.samples[x_range.clone()], &self.labels[y_range])
    }
}
