use super::{dense::Dense, input::Input, sigmoid::Sigmoid};
use ndarray::{Array1, ArrayView1};

pub enum Layer<'a> {
    Input(Input<'a>),
    Dense(Dense<'a>),
    Sigmoid(Sigmoid),
}

impl<'a> Layer<'a> {
    fn forward(&self, x: ArrayView1<f32>) -> Array1<f32> {
        use Layer::*;
        match self {
            Input(l) => l.forward(x),
            Dense(l) => l.forward(x),
            Sigmoid(l) => l.forward(x),
        }
    }
}
