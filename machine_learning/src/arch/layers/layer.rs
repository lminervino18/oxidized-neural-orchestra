use ndarray::Array2;

use super::{Dense, Sigmoid};

pub enum Layer {
    Dense(Dense),
    Sigmoid(Sigmoid),
}
use Layer::*;

impl Layer {
    pub fn dense(dim: (usize, usize)) -> Self {
        Self::Dense(Dense::new(dim))
    }

    pub fn sigmoid(dim: usize, amp: f32) -> Self {
        Self::Sigmoid(Sigmoid::new(dim, amp))
    }

    pub fn forward(&mut self, params: &mut &[f32], x: Array2<f32>) -> Array2<f32> {
        match self {
            Dense(l) => l.forward(params, x),
            Sigmoid(l) => l.forward(x),
        }
    }

    pub fn backward(&mut self, params: &mut &[f32], d: Array2<f32>) -> Array2<f32> {
        match self {
            Dense(l) => l.backward(params, d),
            Sigmoid(l) => l.backward(d),
        }
    }
}
