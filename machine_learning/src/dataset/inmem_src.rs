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
            let x_i_data = i * x_size;
            let y_i_data = i * y_size;
            let x_j_data = j * x_size;
            let y_j_data = j * y_size;

            let (x_left, x_right) = self.samples.split_at_mut(x_j_data);
            let (y_left, y_right) = self.labels.split_at_mut(y_j_data);
            let x_row = &mut x_left[x_i_data..x_i_data + x_size];
            let x_other = &mut x_right[..x_size];
            let y_row = &mut y_left[y_i_data..y_i_data + y_size];
            let y_other = &mut y_right[..y_size];

            x_row.swap_with_slice(x_other);
            y_row.swap_with_slice(y_other);
        }
    }

    pub fn raw_batch(&self, x_range: Range<usize>, y_range: Range<usize>) -> (&[f32], &[f32]) {
        (&self.samples[x_range.clone()], &self.labels[y_range])
    }
}
