use ndarray::{Array2, ArrayView2};

use super::{layers::Layer, loss::LossFn, Model};

pub struct Sequential {
    layers: Vec<Layer>,
}

impl Sequential {
    pub fn new<I>(layers: I) -> Self
    where
        I: IntoIterator<Item = Layer>,
    {
        Self {
            layers: layers.into_iter().collect(),
        }
    }

    pub fn forward(&mut self, mut params: &[f32], mut x: Array2<f32>) -> Array2<f32> {
        for l in self.layers.iter_mut() {
            x = l.forward(&mut params, x);
        }

        x
    }

    // retornar grad_params (grad_w y grad_b)
    pub fn backward<L: LossFn>(
        &mut self,
        mut params: &[f32],
        y_pred: ArrayView2<f32>,
        y: ArrayView2<f32>,
        loss: &L,
    ) {
        let mut d = loss.loss_prime(y_pred, y);

        for l in self.layers.iter_mut().rev() {
            d = l.backward(&mut params, d);
        }
    }
}

impl Model for Sequential {}
