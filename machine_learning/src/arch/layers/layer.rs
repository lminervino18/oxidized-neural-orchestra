use ndarray::prelude::*;

use super::Dense;
use crate::{Result, arch::activations::ActFn};

/// A type of model layer.
#[derive(Clone)]
pub enum Layer {
    Dense(Dense),
}
use Layer::*;

impl Layer {
    /// Returns a new `Layer::Dense` layer.
    ///
    /// # Arguments
    /// * `dim` - The dimension of the layer: (input dimension, and output dimension)
    /// * `act_fn` - The layer's activation function, can be `None` if the output is supposed to be the layer's weighted sums z.
    ///
    /// # Returns
    /// A new `Layer` instance.
    pub fn dense<A: Into<Option<ActFn>>>(dim: (usize, usize), act_fn: A) -> Self {
        Self::Dense(Dense::new(dim, act_fn.into()))
    }

    /// The size of the layer.
    ///
    /// # Returns
    /// The amount of parameters the layer holds.
    pub fn size(&self) -> usize {
        match self {
            Dense(layer) => layer.size(),
        }
    }

    /// Performs a forward pass of the layer and returns a view of its activation.
    ///
    /// # Arguments
    /// * `params` - The parameters to use for the forward pass.
    /// * `x` - The input x that is to be *forwarded*.
    ///
    /// # Returns
    /// The prediction for the given input `x`.
    pub fn forward<'a>(
        &'a mut self,
        params: &[f32],
        x: ArrayView2<f32>,
    ) -> Result<ArrayView2<'a, f32>> {
        match self {
            Dense(layer) => layer.forward(params, x),
        }
    }

    /// Performs a backward pass, writes the gradient of the delta with respect to the layer's portion of
    /// parameters and returns a view of its delta.
    ///
    /// # Arguments
    /// * `params` - The parameters to use for the backward pass.
    /// * `grad` - The buffer for writing the gradient of the layer.
    /// * `d` - The delta of the next layer.
    ///
    /// # Returns
    /// The delta for this layer or an error if occurred.
    pub fn backward<'a>(
        &'a mut self,
        params: &[f32],
        grad: &mut [f32],
        d: ArrayViewMut2<f32>,
    ) -> Result<ArrayViewMut2<'a, f32>> {
        match self {
            Dense(layer) => layer.backward(params, grad, d),
        }
    }
}
