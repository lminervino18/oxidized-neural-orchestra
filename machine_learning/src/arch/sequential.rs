use ndarray::{Array2, ArrayView2, ArrayViewMut2};

use super::{layers::Layer, loss::LossFn, Model};
use crate::dataset::Dataset;

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
        x: ArrayView2<'a, f32>,
    ) -> ArrayView2<'a, f32> {
        let mut output = x;

        for l in self.layers.iter_mut() {
            let curr;
            (curr, params) = params.split_at(l.size());
            output = l.forward(curr, x);
        }

        output
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
    fn backprop<'a, L, I>(
        &mut self,
        params: &mut [f32],
        grad: &mut [f32],
        loss: &L,
        batch: (ArrayView2<f32>, ArrayView2<f32>),
    ) where
        L: LossFn,
        I: IntoIterator<Item = (ArrayView2<'a, f32>, ArrayView2<'a, f32>)>,
    {
        let (x, y) = batch;
        // TODO: cuando esté into iter para dataset usarlo (debería ser como un stream, si no no
        // tiene sentido)
        // TODO: intentar evitar owned. NOTE: si usamos la misma memoria para los cómputos de
        // forward y backward este to_owned pasa a ser algo como un take
        let y_pred = self.forward(params, x).to_owned();
        self.backward(params, grad, y_pred.view(), y, loss);

        // mover los parámetros, deberíamos haber construido de antemano las views para manipular?
        // en ese caso lo que está por debajo podría pasar a recibir arrays también
        // NOTE: los params no se pueden representar como una sóla matriz, por lo tanto la resta va
        // a tener que ser iterando las matrices de cada layer: iterar de nuevo y llamar a
        // apply_grad()
    }
}
