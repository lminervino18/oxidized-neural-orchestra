use rand::Rng;

pub struct InlineSrc {
    samples: Vec<f32>,
    labels: Vec<f32>,
}

impl InlineSrc {
    pub fn new(samples: Vec<f32>, labels: Vec<f32>) -> Self {
        InlineSrc { samples, labels }
    }

    pub fn len(&self) -> usize {
        self.samples.len() + self.labels.len()
    }

    pub fn shuffle<R: Rng>(&mut self, rows: usize, x_size: usize, y_size: usize, rng: &mut R) {
        // TODO: revisar esto porque lo hice quedándome dormido xd
        for i in 0..rows {
            let j = rng.random_range(i..rows);
            if i == j {
                continue;
            }

            let (i, j) = (i.min(j), i.max(j));

            let x_i_data = i * x_size;
            let y_i_data = i * y_size;
            let y_j_data = j * y_size;
            let x_j_data = j * x_size;

            let (x_left, x_right) = self.samples.split_at_mut(x_j_data);
            let (y_left, y_right) = self.labels.split_at_mut(y_j_data);

            let x_row = &mut x_left[x_i_data..x_i_data + x_size];
            let y_row = &mut y_left[y_i_data..y_i_data + y_size];
            let x_other = &mut x_right[..x_size];
            let y_other = &mut y_right[..y_size];

            x_row.swap_with_slice(x_other);
            y_row.swap_with_slice(y_other);
        }
    }

    pub fn raw_batch(
        &self,
        row: usize,
        n: usize,
        x_size: usize,
        y_size: usize,
    ) -> (&[f32], &[f32]) {
        let x_batch = &self.samples[row * x_size..row * x_size + n * x_size];
        let y_batch = &self.labels[row * y_size..row * y_size + n * y_size];

        (x_batch, y_batch)
    }
}
