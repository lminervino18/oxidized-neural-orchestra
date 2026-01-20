use ndarray::{Array1, ArrayView1};
use std::f32;

pub struct Sigmoid {
    dim: usize,
    sigmoid: fn(f32) -> f32,
    sigmoid_prime: fn(f32) -> f32,
}

impl Sigmoid {
    pub fn new(dim: usize) -> Self {
        fn sigmoid(z: f32) -> f32 {
            return 1. / (1. + f32::consts::E.powf(-z));
        }

        fn sigmoid_prime(z: f32) -> f32 {
            let s = sigmoid(z);
            return s * (1. - s);
        }

        Self {
            dim,
            sigmoid,
            sigmoid_prime,
        }
    }

    pub fn forward(&self, x: ArrayView1<f32>) -> Array1<f32> {
        x.mapv(self.sigmoid)
    }
}
