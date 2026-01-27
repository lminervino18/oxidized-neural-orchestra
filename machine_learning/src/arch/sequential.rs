use ndarray::ArrayView2;

use crate::optimization::Optimizer;

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

    pub fn forward<'a>(
        &'a mut self,
        mut params: &[f32],
        mut x: ArrayView2<'a, f32>,
    ) -> ArrayView2<'a, f32> {
        for l in self.layers.iter_mut() {
            let curr;
            (curr, params) = params.split_at(l.size());
            x = l.forward(curr, x);
        }

        x
    }

    pub fn backward<L: LossFn>(
        &mut self,
        params: &[f32],
        grad: &mut [f32],
        y_pred: ArrayView2<f32>,
        y: ArrayView2<f32>,
        loss: &L,
    ) {
        let d_last = loss.loss_prime(y_pred, y);
        let mut d = d_last.view();

        let mut rem_params = params;
        let mut rem_grad = grad;

        for l in self.layers.iter_mut().rev() {
            let curr_params;
            let curr_grad;
            (rem_params, curr_params) = rem_params.split_at(l.size());
            (rem_grad, curr_grad) = rem_grad.split_at_mut(l.size());

            d = l.backward(curr_params, curr_grad, d.view());
        }
    }
}

impl Model for Sequential {
    fn size(&self) -> usize {
        self.layers.iter().map(|layer| layer.size()).sum()
    }

    fn backprop<'a, L, O, I>(
        &mut self,
        params: &mut [f32],
        grad: &mut [f32],
        loss: &L,
        optimizer: &mut O,
        batches: I,
    ) where
        L: LossFn,
        O: Optimizer,
        I: Iterator<Item = (ArrayView2<'a, f32>, ArrayView2<'a, f32>)>,
    {
        for (i, (x, y)) in batches.enumerate() {
            if i > 0 {
                optimizer.update_params(params, grad);
            }

            // TODO: sacar `to_owned`
            let y_pred = self.forward(params, x).to_owned();
            self.backward(params, grad, y_pred.view(), y, loss);
        }
    }
}
