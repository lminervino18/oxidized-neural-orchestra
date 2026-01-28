use ndarray::ArrayView2;

use crate::optimization::Optimizer;

use super::{layers::Layer, loss::LossFn, Model};

#[derive(Clone)]
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
            let (curr, rest) = params.split_at(l.size());
            (x, params) = (l.forward(curr, x), rest);
        }

        x
    }

    pub fn backward<L: LossFn>(
        &mut self,
        mut params: &[f32],
        mut grad: &mut [f32],
        y_pred: ArrayView2<f32>,
        y: ArrayView2<f32>,
        loss: &L,
    ) {
        let mut d_last = loss.loss_prime(y_pred, y);
        let mut d = d_last.view_mut();

        for l in self.layers.iter_mut().rev() {
            let (ps_rest, ps_curr) = params.split_at(params.len() - l.size());
            let (gs_rest, gs_curr) = grad.split_at_mut(grad.len() - l.size());
            (d, params, grad) = (l.backward(ps_curr, gs_curr, d.view_mut()), ps_rest, gs_rest);
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
        for (x, y) in batches {
            // TODO: sacar `to_owned`
            grad.fill(0.0);
            let y_pred = self.forward(params, x).to_owned();

            let err = loss.loss(y_pred.view(), y);
            println!("err: {}", err);

            self.backward(params, grad, y_pred.view(), y, loss);
            optimizer.update_params(params, grad);
        }
    }
}
