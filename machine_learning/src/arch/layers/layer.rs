use ndarray::{Data, RawData, prelude::*};

use super::{Conv2d, Dense, Sigmoid};
use crate::{MlErr, Result, arch::layers::Reshape};

/// An indirection layer to prevent leaking the
/// inner enum representation to the upper mods.
#[derive(Clone)]
enum Inner {
    Conv2d(Conv2d),
    Dense(Dense),
    Sigmoid(Sigmoid),
    Reshape(Reshape),
}
use Inner::*;

/// Represents the different types of layers in a model.
#[derive(Clone)]
pub struct Layer(Inner);

impl Layer {
    /// Creates a new `Layer::Dense` layer.
    ///
    /// # Args
    /// * `dim` - The dimension of the layer: (input dimension, and output dimension)
    ///
    /// # Returns
    /// A new `Layer` instance.
    pub fn dense(dim: (usize, usize)) -> Self {
        Self(Inner::Dense(Dense::new(dim)))
    }

    /// Creates a new `Layer::Sigmoid` layer.
    ///
    /// # Args
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
    /// * `kernel_size` - The height/width of the square kernel.
    /// * `stride` - The stride for the kernel.
    /// * `padding` - The parameter padding size.
    ///
    /// # Returns
    /// A new `Layer` instance.
    pub fn conv2d(
        filters: usize,
        in_channels: usize,
        kernel_size: usize,
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

    /// Creates a new `Layer::Reshape` layer that reshapes 2D tensors into 4D ones.
    ///
    /// # Arguments
    /// * `channels` - The amount of input channels.
    /// * `height` - The height and width of the matrices.
    /// * `width` - The width and width of the matrices.
    ///
    /// # Returns
    /// A new `Layer` instance.
    pub fn two_d_to4d(channels: usize, height: usize, width: usize) -> Self {
        Self(Inner::Reshape(Reshape::two_d_to4d(channels, height, width)))
    }

    /// Creates a new `Layer::Reshape` layer that reshapes 4D tensors into 2D ones.
    ///
    /// # Arguments
    /// * `channels` - The amount of input channels.
    /// * `height` - The height and width of the matrices.
    /// * `width` - The width and width of the matrices.
    ///
    /// # Returns
    /// A new `Layer` instance.
    pub fn four_d_to2d(channels: usize, height: usize, width: usize) -> Self {
        Self(Inner::Reshape(Reshape::four_d_to2d(
            channels, height, width,
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
            Reshape(layer) => layer.size(),
        }
    }

    /// Performs a forward pass of the layer and returns a view of its activation.
    ///
    /// # Args
    /// * `params` - The parameters to use for the forward pass.
    /// * `x` - The input x that is to be *forwarded*.
    ///
    /// # Returns
    /// The prediction for the given input `x`.
    pub fn forward<'a>(
        &'a mut self,
        params: &[f32],
        x: ArrayViewD<'a, f32>,
    ) -> Result<ArrayViewD<'a, f32>> {
        let y = match &mut self.0 {
            Conv2d(layer) => layer.forward(params, try_cast_dim(x)?)?.into_dyn(),
            Dense(layer) => layer.forward(params, try_cast_dim(x)?)?.into_dyn(),
            Sigmoid(layer) => layer.forward(try_cast_dim(x)?)?.into_dyn(),
            Reshape(layer) => layer.forward(x)?,
        };

        Ok(y)
    }

    /// Performs a backward pass, writes the gradient of the delta with respect to the layer's portion of
    /// parameters and returns a view of its delta.
    ///
    /// # Args
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
        d: ArrayViewMutD<'a, f32>,
    ) -> Result<ArrayViewMutD<'a, f32>> {
        let q = match &mut self.0 {
            Conv2d(layer) => layer.backward(params, grad, try_cast_dim(d)?)?.into_dyn(),
            Dense(layer) => layer.backward(params, grad, try_cast_dim(d)?)?.into_dyn(),
            Sigmoid(layer) => layer.backward(try_cast_dim(d)?)?.into_dyn(),
            Reshape(layer) => layer.backward(try_cast_dim(d)?)?,
        };

        Ok(q)
    }
}

/// Tries to cast the dynamic dimension shape of the given array into a concrete dimension.
///
/// # Arguments
/// * `arr` - The array to redimension.
///
/// # Errors
/// A `MlErr::DimMismatch` if the dimension of the array and the required dimension don't agree.
///
/// # Returns
/// The newly redimension array.
fn try_cast_dim<S, D>(arr: ArrayBase<S, IxDyn>) -> Result<ArrayBase<S, D>>
where
    S: RawData<Elem = f32> + Data,
    D: Dimension,
{
    arr.into_dimensionality().map_err(|e| MlErr::MatrixError(e))
}
