use ndarray::{ArrayView2, ArrayViewMut2};

use super::{layers::Layer, loss::LossFn};
use crate::{MlErr, Result, optimization::Optimizer, param_manager::ParamManager};

/// A trainable model, this model's architecture is a sequence of trainable layers.
#[derive(Clone)]
pub struct Sequential {
    layers: Vec<Layer>,
}

impl Sequential {
    /// Creates a new `Sequential`.
    ///
    /// # Arguments
    /// * `layers` - A list of layers.
    ///
    /// # Returns
    /// A new `Sequential` instance.
    pub fn new(layers: Vec<Layer>) -> Self {
        Self { layers }
    }

    /// Calculates the amount of parameters in the model.
    ///
    /// # Returns
    /// The size of this model in the amount of parameters.
    pub fn size(&self) -> usize {
        self.layers.iter().map(|layer| layer.size()).sum()
    }

    /// Makes a forward pass through the network.
    ///
    /// # Arguments
    /// * `param_manager` - The manager of parameters.
    /// * `x` - The input data.
    ///
    /// # Errors
    /// An error if there's a size mismatch between the layers' sizes and the parameter manager
    ///
    /// # Returns
    /// The prediction for the given input or an error if occurred.
    pub fn forward<'x, 'mw>(
        &'x mut self,
        param_manager: &mut ParamManager<'mw>,
        x: ArrayView2<'x, f32>,
    ) -> Result<ArrayView2<'x, f32>> {
        let mut front = param_manager.front();
        let n = self.layers.len();
        let mut x = x.into_dyn();

        for (i, layer) in self.layers.iter_mut().enumerate() {
            let params = front.next(layer.size()).ok_or(MlErr::SizeMismatch {
                what: "layers",
                got: i,
                expected: n,
            })?;

            x = layer.forward(params, x)?;
        }

        // SAFETY: For now just support 2d outputs.
        Ok(x.into_dimensionality().unwrap())
    }

    /// Makes a backward pass through the network.
    ///
    /// # Arguments
    /// * `param_manager` - The manager of parameters.
    /// * `d` - The starting delta, the loss prime.
    ///
    /// # Errors
    /// An error if there's a size mismatch between the layers' sizes and the parameter manager
    ///
    /// # Returns
    /// An error if occurred.
    pub fn backward<'d, 'mw>(
        &'d mut self,
        param_manager: &mut ParamManager<'mw>,
        d: ArrayViewMut2<'d, f32>,
    ) -> Result<()> {
        let mut back = param_manager.back();
        let n = self.layers.len();
        let mut d = d.into_dyn();

        for (i, layer) in self.layers.iter_mut().rev().enumerate() {
            let (params, grad) = back.next(layer.size()).ok_or(MlErr::SizeMismatch {
                what: "layers",
                got: i,
                expected: n,
            })?;

            d = layer.backward(params, grad, d)?;
        }

        Ok(())
    }

    /// Computes the gradient of the loss function with respect to the parameters of the model over
    /// the provided batches. **`params` gets updated** for each batch according to the
    /// optimization algorithm.
    ///
    /// Since getting the actual loss would require forwarding over all batches again at
    /// the end of the backprop iterations, we are approximating it by averaging the loss at
    /// each batch (this is a good approximation), another option would be to sum the weighted
    /// losses, that is, loss * batch_size and then diving by the also weighted sum of
    /// num_batches.
    ///
    /// # Arguments
    /// * `params` - The model's parameters.
    /// * `grad` - A buffer for writing the computed gradient on each batch pass.
    /// * `loss` - The loss function.
    /// * `optimizer` - The optimizer that dictates how to update the weights on each gradient calculation.
    /// * `batches` - The batches of data.
    ///
    /// # Errors
    /// An error if there's a size mismatch between the layers' sizes and the parameter manager
    ///
    /// # Returns
    /// The epoch loss or an error if the model failed to run a backpropagation epoch.
    pub fn backprop<'a, 'mw, O, L, I>(
        &mut self,
        param_manager: &mut ParamManager<'mw>,
        optimizers: &mut [O],
        loss_fn: &mut L,
        batches: I,
    ) -> Result<f32>
    where
        L: LossFn,
        O: Optimizer + Send,
        I: Iterator<Item = (ArrayView2<'a, f32>, ArrayView2<'a, f32>)>,
    {
        let mut total_loss = 0.0;
        let mut num_batches = 0;

        for (x, y) in batches {
            let y_pred = self.forward(param_manager, x)?;
            let (loss, mut d) = loss_fn.loss_prime(y_pred, y);

            total_loss += loss;
            num_batches += 1;

            self.backward(param_manager, d.view_mut())?;

            param_manager.optimize(optimizers)?;
            param_manager.acc_grad();
            param_manager.zero_grad();
        }

        Ok(total_loss / num_batches as f32)
    }
}
