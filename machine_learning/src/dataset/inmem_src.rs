use rand::Rng;
use std::ops::Range;

pub struct InMemSrc {
    data: Vec<f32>,
}

impl InMemSrc {
    pub fn new(data: Vec<f32>) -> Self {
        InMemSrc { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn shuffle<R: Rng>(&mut self, rows: usize, row_size: usize, rng: &mut R) {
        for i in 0..rows {
            let j = rng.random_range(i..rows);
            if i == j {
                continue;
            }

            let (i, j) = (i.min(j), i.max(j));
            let i_data = i * row_size;
            let j_data = j * row_size;

            let (left, right) = self.data.split_at_mut(j_data);
            let row = &mut left[i_data..i_data + row_size];
            let other = &mut right[..row_size];
            row.swap_with_slice(other);
        }
    }

    pub fn raw_batch(&self, range: Range<usize>) -> &[f32] {
        &self.data[range]
    }
}
