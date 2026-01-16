use ndarray::{Array1, ArrayView1};

pub struct Dataset {
    x_train: Vec<Array1<f32>>,
    y_train: Vec<Array1<f32>>,
}

/* impl Iterator for Dataset {
    type Item = (ArrayView1<f32>, ArrayView1<f32>);

    fn next(&mut self) -> Option<Self::Item> {}
} */
