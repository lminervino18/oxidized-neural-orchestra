use ndarray::{ArrayView2, Axis};

pub struct Dataset {
    data: Vec<f32>,
    len: usize,
    x_size: usize,
    y_size: usize,
}

impl Dataset {
    pub fn new(data: Vec<f32>, x_size: usize, y_size: usize) -> Self {
        Self {
            len: data.len(),
            data,
            x_size,
            y_size,
        }
    }

    pub fn get(&self, row_offset: usize, n_rows: usize) -> (ArrayView2<f32>, ArrayView2<f32>) {
        let &Self {
            x_size,
            y_size,
            ref data,
            ..
        } = self;

        let row_size = x_size + y_size;
        let offset = row_offset * row_size;
        let raw_batch = &data[offset..offset + row_size * n_rows];

        let batch = ArrayView2::from_shape((n_rows, row_size), raw_batch).unwrap();
        let (x, y) = batch.split_at(Axis(1), x_size);

        (x, y)
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

        let (x, y) = ds.get(0, 2);

        assert_eq!(x, expected_x);
        assert_eq!(y, expected_y);
    }
}
