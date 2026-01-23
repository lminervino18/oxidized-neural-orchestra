use ndarray::{Array2, ArrayView2, ArrayViewMut2};
use std::f32;

pub struct Sigmoid {
    amp: f32,
    dim: usize,
}

impl Sigmoid {
    pub fn new(dim: usize, amp: f32) -> Self {
        // fn sigmoid_prime(z: f32) -> f32 {
        //     let s = sigmoid(z);
        //     return s * (1. - s);
        // }

        Self { dim, amp }
    }

    fn sigmoid(&self, z: f32) -> f32 {
        let s = 1. / (1. + f32::consts::E.powf(-z));
        s * self.amp
    }

    fn sigmoid_prime(&self, z: f32) -> f32 {
        let s = self.sigmoid(z);
        s * (1. - s)
    }

    pub fn forward(&self, x: ArrayView2<f32>) -> Array2<f32> {
        x.mapv(|z| self.sigmoid(z))
    }

    pub fn backward(&mut self, d: ArrayViewMut2<f32>) -> Array2<f32> {
        // d.mapv_into(|z| self.sigmoid_prime(z))
        todo!()
    }
}
