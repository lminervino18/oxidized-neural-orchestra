use ndarray::ArrayView2;

use super::{Model, layers::Layer, loss::LossFn};
use crate::{MlErr, Result, middleware::ParamManager, optimization::Optimizer};

/// A sequential model: information flows forward when computing an output and backward when
/// computing the *deltas* of its layers.
#[derive(Clone)]
pub struct Sequential {
    layers: Vec<Layer>,
}

impl Sequential {
    /// Creates a new `Sequential`.
    ///
    /// # Arguments
    /// * `layers` - The layers the sequential is composed of.
    ///
    /// # Returns
    /// A new `Sequential` instance.
    pub fn new<I>(layers: I) -> Self
    where
        I: IntoIterator<Item = Layer>,
    {
        Self {
            layers: layers.into_iter().collect(),
        }
    }

    /// Makes a forward pass through the network.
    ///
    /// # Arguments
    /// * `param_manager` - The manager of parameters.
    /// * `x` - The input data.
    ///
    /// # Returns
    /// The prediction for the given input or an error if occurred.
    pub fn forward<'x, 'mw>(
        &'x mut self,
        param_manager: &mut ParamManager<'mw>,
        mut x: ArrayView2<'x, f32>,
    ) -> Result<ArrayView2<'x, f32>> {
        let mut front = param_manager.front();
        let nlayers = self.layers.len();

        for (i, layer) in self.layers.iter_mut().enumerate() {
            let size = layer.size();
            let params = front.next(size).ok_or_else(|| MlErr::SizeMismatch {
                what: "layers",
                got: i,
                expected: nlayers,
            })?;

            x = layer.forward(params, x)?;
        }

        Ok(x)
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
    fn backprop<'a, 'mw, O, L, I>(
        &mut self,
        param_manager: &mut ParamManager<'mw>,
        optimizers: &mut [O],
        loss_fn: &L,
        batches: I,
    ) -> Result<f32>
    where
        L: LossFn,
        O: Optimizer + Send,
        I: Iterator<Item = (ArrayView2<'a, f32>, ArrayView2<'a, f32>)>,
    {
        let nlayers = self.layers.len();
        let mut total_loss = 0.0;
        let mut num_batches = 0;

        for (x, y) in batches {
            param_manager.zero_grad();

            let y_pred = self.forward(param_manager, x)?;
            total_loss += loss_fn.loss(y_pred, y);
            num_batches += 1;

            let mut back = param_manager.back();
            let mut d_last = loss_fn.loss_prime(y_pred, y);
            let mut d = d_last.view_mut();

            for (i, layer) in self.layers.iter_mut().rev().enumerate() {
                let size = layer.size();
                let (params, grad) = back.next(size).ok_or_else(|| MlErr::SizeMismatch {
                    what: "layers",
                    got: i,
                    expected: nlayers,
                })?;

                d = layer.backward(params, grad, d)?;
            }

            param_manager.optimize(optimizers)?;
        }

        Ok(total_loss / num_batches as f32)
    }
}
