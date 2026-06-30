use std::cmp;

use ndarray::prelude::*;
use ndarray_conv::{ConvMode, PaddingMode};

use super::Convolver;
use crate::{MlErr, Result, arch::InplaceReshape};

#[derive(Clone, Debug)]
pub struct Conv2d {
    filters: usize,
    in_channels: usize,
    /// The size of the square kernel matrix.
    kernel_size: usize,
    stride: usize,
    padding: usize,

    kernels_size: usize,
    /// The dimension of the kernels tensor, `(filters, in_channels, kernel_size,
    /// kernel_size)`
    kernels_dim: (usize, usize, usize, usize),
    size: usize,

    // Forward metadata
    real_input_dim: (usize, usize),
    /// The input that's actually used during the forward convolution
    effective_input: Array4<f32>,
    output: Array4<f32>,

    // Backward metadata
    delta_out: Array4<f32>,

    convolver: Convolver,
}

impl Conv2d {
    pub fn new(
        filters: usize,
        in_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
        let kernels_size = filters * in_channels * kernel_size * kernel_size;
        let kernels_dim = (filters, in_channels, kernel_size, kernel_size);
        let size = kernels_size + filters;
        let real_input_dim = (0, 0);

        let zeros4 = Array4::zeros((1, 1, 1, 1));

        let convolver = Convolver::new(kernel_size, None);

        Self {
            filters,
            in_channels,
            kernel_size,
            stride,
            padding,
            kernels_size,
            kernels_dim,
            size,
            real_input_dim,
            effective_input: zeros4.clone(),
            output: zeros4.clone(),
            delta_out: zeros4.clone(),
            convolver,
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn forward(
        &mut self,
        params: &[f32],
        input: ArrayView4<f32>,
    ) -> Result<ArrayView4<'_, f32>> {
        let (kernel, bias) = self.view_params(params)?;

        let Self {
            filters,
            kernel_size,
            stride,
            padding,
            ref mut real_input_dim,
            ref mut effective_input,
            ref mut output,
            ref mut convolver,
            ..
        } = *self;

        let (batch_size, _, input_h, input_w) = input.dim();

        *real_input_dim = (input_h, input_w);

        let out_h = (input_h + 2 * padding - kernel_size) / stride + 1;
        let out_w = (input_w + 2 * padding - kernel_size) / stride + 1;

        let effective_h = (out_h - 1) * stride + kernel_size;
        let effective_w = (out_w - 1) * stride + kernel_size;

        effective_input.reshape_inplace((input.dim().0, input.dim().1, effective_h, effective_w));
        effective_input.fill(0.);

        // dropped elements could just be padding
        let copy_h = cmp::min(input_h, effective_h - padding);
        let copy_w = cmp::min(input_w, effective_w - padding);

        let mut effective_input_view = effective_input.slice_mut(s![
            ..,
            ..,
            padding..padding + copy_h,
            padding..padding + copy_w,
        ]);
        let input_view = &input.slice(s![.., .., ..copy_h, ..copy_w]);
        effective_input_view.assign(input_view);

        output.reshape_inplace((batch_size, filters, out_h, out_w));

        effective_input
            .axis_iter(Axis(0))
            .zip(output.axis_iter_mut(Axis(0)))
            .for_each(|(input_b, mut output_b)| {
                convolver.conv_im2col_into(&mut output_b, input_b, kernel, false, stride);
            });

        *output += &bias;

        Ok(output.view())
    }

    pub fn backward(
        &mut self,
        params: &[f32],
        grad: &mut [f32],
        delta_in: ArrayViewMut4<f32>,
    ) -> Result<ArrayViewMut4<'_, f32>> {
        let (mut d_kernel, mut d_bias) = self.view_grad(grad)?;
        let (kernel, _) = self.view_params(params)?;

        let Self {
            filters,
            in_channels,
            kernel_size,
            stride,
            padding,
            real_input_dim,
            ref effective_input,
            ref mut delta_out,
            ref mut convolver,
            ..
        } = *self;

        d_kernel.fill(0.);
        delta_out.reshape_inplace((
            effective_input.dim().0,
            effective_input.dim().1,
            real_input_dim.0,
            real_input_dim.1,
        ));
        delta_out.fill(0.);

        delta_in
            .axis_iter(Axis(0))
            .zip(effective_input.axis_iter(Axis(0)))
            .zip(delta_out.axis_iter_mut(Axis(0)))
            .for_each(|((delta_in_b, effective_input_b), mut delta_out_b)| {
                convolver.conv_col2im_into(&mut delta_out_b, delta_in_b, kernel, false, stride);
                // let d_kernel2 = effective_input
            });

        let db_sum = delta_in
            .sum_axis(Axis(0))
            .sum_axis(Axis(1))
            .sum_axis(Axis(1));
        d_bias.assign(&db_sum);

        Ok(delta_out.view_mut())
    }

    /// Gives a view of the raw parameter slice as the weights and biases of this layer.
    ///
    /// # Arguments
    /// * `params` - A slice of parameters.
    ///
    /// # Returns
    /// A tuple containing the weights and biases or an error if there's a mismatch
    /// between the size of the gradient and the size of the layer.
    fn view_params<'a>(
        &self,
        params: &'a [f32],
    ) -> Result<(ArrayView4<'a, f32>, ArrayView4<'a, f32>)> {
        let Self {
            filters,
            kernels_size,
            kernels_dim,
            size,
            ..
        } = *self;

        if params.len() != size {
            return Err(MlErr::size_mismatch("params", params.len(), size));
        }

        // SAFETY: The if condition above checks that the size of the
        //         parameters is exactly the size of the layer.
        let weights = ArrayView4::from_shape(kernels_dim, &params[..kernels_size]).unwrap();
        let biases = ArrayView4::from_shape((1, filters, 1, 1), &params[kernels_size..]).unwrap();

        Ok((weights, biases))
    }

    /// Gives a view of the raw gradient slice as the delta weights and delta biases of this layer.
    ///
    /// # Arguments
    /// * `grad` - A gradient slice.
    ///
    /// # Returns
    /// A tuple containing the delta weights and delta biases or an error if there's
    /// a mismatch between the size of the gradient and the size of the layer.
    fn view_grad<'a>(
        &self,
        grad: &'a mut [f32],
    ) -> Result<(ArrayViewMut4<'a, f32>, ArrayViewMut1<'a, f32>)> {
        let Self {
            filters,
            kernels_size,
            kernels_dim,
            size,
            ..
        } = *self;

        if grad.len() != size {
            return Err(MlErr::size_mismatch("grad", grad.len(), self.size));
        }

        // SAFETY: The if condition above checks that the size of the
        //         gradient is exactly the size of the layer.
        let (dw_raw, db_raw) = grad.split_at_mut(kernels_size);

        let dw = ArrayViewMut4::from_shape(kernels_dim, dw_raw).unwrap();
        let db = ArrayViewMut1::from_shape(filters, db_raw).unwrap();

        Ok((dw, db))
    }
}
