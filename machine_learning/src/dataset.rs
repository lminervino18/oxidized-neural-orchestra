use ndarray::{s, ArrayView2};

pub struct Dataset {
    x_size: usize,
    y_size: usize,
    len: usize,
    data: Vec<f32>,
}

impl Dataset {
    pub fn get(&self, row: usize, amount: usize) -> (ArrayView2<f32>, ArrayView2<f32>) {
        let &Self {
            x_size,
            y_size,
            len,
            ref data,
        } = self;

        let full_view = ArrayView2::from_shape((x_size + y_size, len), data).unwrap();
        let x = full_view.slice(s![.., 0..self.x_size]);
        let y = full_view.slice(s![.., 0..self.x_size]);

        // (x, y)
        todo!()
    }
}
