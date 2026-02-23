use ndarray::ArrayView2;

use crate::{middleware::ParamManager, optimization::Optimizer};

use super::{Model, layers::Layer, loss::LossFn};

/// A sequential model: information flows forward when computing an output and backward when
/// computing the *deltas* of its layers.
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

    /// Makes a forward pass through the network.
    ///
    /// # Arguments
    /// * `param_manager` - The manager of parameters.
    /// * `x` - The input data.
    ///
    /// # Returns
    /// The prediction for the given input.
    pub fn forward<'x, 'mw>(
        &'x mut self,
        param_manager: &mut ParamManager<'mw>,
        mut x: ArrayView2<'x, f32>,
    ) -> Option<ArrayView2<'x, f32>> {
        let mut front = param_manager.front();

        for layer in self.layers.iter_mut() {
            let params = front.next()?;
            x = layer.forward(params, x);
        }

        Some(x)
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
    ) -> f32
    where
        L: LossFn,
        O: Optimizer,
        I: Iterator<Item = (ArrayView2<'a, f32>, ArrayView2<'a, f32>)>,
    {
        let mut total_loss = 0.0;
        let mut num_batches = 0;

        for (mut x, y) in batches {
            param_manager.zero_grad();

            // TODO: Ver como justificar este unwrap o devolver un error
            x = self.forward(param_manager, x).unwrap();

            total_loss += loss_fn.loss(x, y);
            num_batches += 1;

            let mut back = param_manager.back();
            let mut d_last = loss_fn.loss_prime(x, y);
            let mut d = d_last.view_mut();

            for layer in self.layers.iter_mut().rev() {
                // TODO: Ver como justificar este unwrap o devolver un error
                let (params, grad) = back.next().unwrap();
                d = layer.backward(params, grad, d.view_mut());
            }

            param_manager.optimize(optimizers);
        }

        total_loss / num_batches as f32
    }
}
