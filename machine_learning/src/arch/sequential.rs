use ndarray::ArrayView2;

use super::{Model, layers::Layer, loss::LossFn};
use crate::{
    error::{MlErr, Result},
    optimization::Optimizer,
    training::{BackIter, FrontIter, ParamManager},
};

/// A sequential model: information flows forward when computing an output and backward when
/// computing the *deltas* of its layers.
#[derive(Clone)]
pub struct Sequential {
    layers: Vec<Layer>,
}

impl Sequential {
    /// Returns a new `Sequential`.
    ///
    /// # Arguments
    /// * `layers` - The layers the sequential is composed of.
    pub fn new<I>(layers: I) -> Self
    where
        I: IntoIterator<Item = Layer>,
    {
        Self {
            layers: layers.into_iter().collect(),
        }
    }

    /// Goes through the model's layers, computes all of their outputs and returns a view of the
    /// output of the last one.
    ///
    /// # Arguments
    /// * `params` - A filled parameter manager.
    /// * `x` - The x batch in the input layer.
    ///
    /// # Returns
    /// The model's prediction for the given input.
    pub fn forward<'a>(
        &'a mut self,
        mut params: FrontIter<'a>,
        mut x: ArrayView2<'a, f32>,
    ) -> Result<ArrayView2<'a, f32>> {
        let n = self.layers.len();

        for (i, layer) in self.layers.iter_mut().enumerate() {
            let size = layer.size();
            let curr = params.take(size).ok_or_else(|| MlErr::SizeMismatch {
                a: "the amount of slices in the parameter manager",
                b: "the amount of layers of the model",
                got: i,
                expected: n,
            })?;

            x = layer.forward(curr, x);
        }

        Ok(x)
    }

    fn backward<'a, L: LossFn>(
        &mut self,
        mut params: BackIter<'a>,
        mut grad: &mut [f32],
        y_pred: ArrayView2<f32>,
        y: ArrayView2<f32>,
        loss: &L,
    ) -> Result<()> {
        let mut d_last = loss.loss_prime(y_pred, y);
        let mut d = d_last.view_mut();
        let n = self.layers.len();

        for (i, layer) in self.layers.iter_mut().rev().enumerate() {
            let size = layer.size();
            let curr = params.take(size).ok_or_else(|| MlErr::SizeMismatch {
                a: "the amount of slices in the parameter manager",
                b: "the amount of layers of the model",
                got: i,
                expected: n,
            })?;

            let (gs_rest, gs_curr) = grad.split_at_mut(grad.len() - size);
            (d, grad) = (layer.backward(curr, gs_curr, d.view_mut()), gs_rest);
        }

        Ok(())
    }
}

impl Model for Sequential {
    fn size(&self) -> usize {
        self.layers.iter().map(|layer| layer.size()).sum()
    }

    // NOTE: since getting the actual loss would require forwarding over all batches again at
    // the end of the backprop iterations, we are approximating it by averaging the loss at
    // each batch (this is a good approximation), another option would be to sum the weighted
    // losses, that is, loss * batch_size and then diving by the also weighted sum of
    // num_batches.
    fn backprop<'a, 'b, L, O, I>(
        &mut self,
        params: &mut ParamManager<'b>,
        grad: &mut [f32],
        loss: &L,
        optimizer: &mut O,
        batches: I,
    ) -> Result<f32>
    where
        L: LossFn,
        O: Optimizer,
        I: Iterator<Item = (ArrayView2<'a, f32>, ArrayView2<'a, f32>)>,
    {
        let mut total_loss = 0f64;
        let mut num_batches = 0usize;

        for (x, y) in batches {
            grad.fill(0.0);

            // TODO: sacar `to_owned`
            let y_pred = self.forward(params.front()?, x)?.to_owned();
            self.backward(params.back()?, grad, y_pred.view(), y, loss)?;
            total_loss += loss.loss(y_pred.view(), y) as f64;
            num_batches += 1;

            let mut front = params.front()?;
            let mut gs_rest = &*grad;
            let slices: Vec<_> = self
                .layers
                .iter()
                .map(|layer| {
                    let size = layer.size();

                    // SAFETY: Previously the forward pass was successful
                    //         with these same sizes in this same order.
                    let slice = front.take(size).unwrap();
                    let (curr, rest) = grad.split_at(size);
                    gs_rest = rest;
                    (slice, curr)
                })
                .collect();

            // TODO: Esto se puede paralelizar, habría que pasar a tener un optimizer por
            //       servidor/capa, cosa que podamos tener optimizadores statefull, estilo Adam.
            for (params, grad) in slices {
                optimizer.update_params(params, grad);
            }
        }

        Ok((total_loss / num_batches as f64) as f32)
    }
}
