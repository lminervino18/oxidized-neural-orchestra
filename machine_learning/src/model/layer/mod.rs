mod layers;

use layers::{Dense, Sigmoid};
use ndarray::{Array2, ArrayView2};

pub enum Layer {
    Dense(Dense),
    Sigmoid(Sigmoid),
}
use Layer::*;

impl Layer {
    pub fn forward(&mut self, params: &mut &[f32], x: ArrayView2<f32>) -> Array2<f32> {
        match self {
            Dense(l) => l.forward(params, x),
            Sigmoid(l) => l.forward(x),
        }
    }
}
