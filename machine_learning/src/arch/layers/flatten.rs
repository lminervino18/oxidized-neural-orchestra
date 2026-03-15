use ndarray::{ArrayView2, ArrayViewD};

use crate::Result;

#[derive(Clone)]
pub struct Flatten {
    input_shape: Vec<usize>,
}

impl Flatten {
    pub fn new() -> Self {
        Self {
            input_shape: Vec::new(),
        }
    }

    pub fn forward<'a>(&mut self, x: ArrayViewD<'a, f32>) -> Result<ArrayView2<'a, f32>> {
        todo!()
    }
}
