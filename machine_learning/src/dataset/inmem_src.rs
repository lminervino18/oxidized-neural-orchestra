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

#[cfg(test)]
mod tests {
    use rand::{SeedableRng, rngs::StdRng};

    use super::*;

    // agrego este test porq me quedé pensando si no había roto algo xd
    #[test]
    fn some_test() {
        let rows = 5;
        let x_size = 2;
        let y_size = 1;
        let samples = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10.];
        let labels = vec![11., 12., 13., 14., 15.];
        let mut rng = StdRng::seed_from_u64(42);

        let mut in_mem = InMemSrc { samples, labels };
        in_mem.shuffle(rows, x_size, y_size, &mut rng);

        // antes del refactor
        let samples_before = [1.0, 2.0, 7.0, 8.0, 5.0, 6.0, 9.0, 10.0, 3.0, 4.0];
        let labels_before = [11.0, 14.0, 13.0, 15.0, 12.0];

        let samples_now = in_mem.samples;
        let labels_now = in_mem.labels;

        assert_eq!(samples_now, samples_before);
        assert_eq!(labels_now, labels_before);
    }
}
