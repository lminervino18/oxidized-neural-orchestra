use ndarray::prelude::*;

use super::{Conv2d, Dense, Sigmoid};
use crate::Result;

mod private {
    use super::*;

    /// An indirection layer to prevent leaking the
    /// inner enum representation to the upper mods.
    #[derive(Clone)]
    pub(super) enum Inner {
        Dense(Dense),
        Sigmoid(Sigmoid),
        Conv2d(Conv2d),
    }
}
use private::Inner::{self, *};

/// Represents the different types of layers in a model.
#[derive(Clone)]
pub struct Layer(Inner);

impl Layer {
    /// Creates a new `Layer::Dense` layer.
    ///
    /// # Arguments
    /// * `dim` - The dimension of the layer: (input dimension, and output dimension)
    ///
    /// # Returns
    /// A new `Layer` instance.
    pub fn dense(dim: (usize, usize)) -> Self {
        Self(Inner::Dense(Dense::new(dim)))
    }

    /// Creates a new `Layer::Sigmoid` layer.
    ///
    /// # Arguments
    /// * `amp` - The amlitude of the sigmoid.
    ///
    /// # Returns
    /// A new `Layer` instance.
    pub fn sigmoid(amp: f32) -> Self {
        Self(Inner::Sigmoid(Sigmoid::new(amp)))
    }

    /// Creates a new `Layer::Conv2d` layer.
    ///
    /// # Arguments
    /// * `filters` - The amount of filters.
    /// * `in_channels` - The amount of input channels.
    /// * `kernel_size` - The height and width of the kernel.
    /// * `stride` - The stride for the kernel.
    /// * `padding` - The parameter padding size.
    ///
    /// # Returns
    /// A new `Layer` instance.
    pub fn conv2d(
        filters: usize,
        in_channels: usize,
        kernel_size: (usize, usize),
        stride: usize,
        padding: usize,
    ) -> Self {
        Self(Inner::Conv2d(Conv2d::new(
            filters,
            in_channels,
            kernel_size,
            stride,
            padding,
        )))
    }

    /// The size of the layer.
    ///
    /// # Returns
    /// The amount of parameters the layer holds.
    pub fn size(&self) -> usize {
        match &self.0 {
            Dense(layer) => layer.size(),
            Sigmoid(layer) => layer.size(),
            Conv2d(layer) => layer.size(),
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
        match &mut self.0 {
            Dense(layer) => layer.forward(params, x),
            Sigmoid(layer) => layer.forward(x),
            _ => todo!(),
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
        d: ArrayViewMut2<'a, f32>,
    ) -> Result<ArrayViewMut2<'a, f32>> {
        match &mut self.0 {
            Dense(layer) => layer.backward(params, grad, d),
            Sigmoid(layer) => layer.backward(d),
            _ => todo!(),
        }
    }
}
