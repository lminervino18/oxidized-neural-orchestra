use std::cmp;

use ndarray::{linalg, prelude::*};

use crate::{MlErr, Result, arch::InplaceReshape};

#[derive(Clone, Debug)]
pub struct Conv2d {
    /// The shape of the kernel tensor, `(filters, in_channels, kernel_size, kernel_size)`.
    kernel_shape: (usize, usize, usize, usize),
    stride: usize,
    padding: usize,

    // Forward metadata
    real_input_shape: (usize, usize, usize, usize),
    /// The input that's actually used during the forward convolution
    effective_input: Array4<f32>,
    output: Array4<f32>,

    // Backward metadata
    delta_out: Array4<f32>,
}

impl Conv2d {
    pub fn new(
        filters: usize,
        in_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
        let kernel_shape = (filters, in_channels, kernel_size, kernel_size);
        let real_input_shape = (0, 0, 0, 0);

        let zeros4 = Array4::zeros((1, 1, 1, 1));

        Self {
            stride,
            padding,
            kernel_shape,
            real_input_shape,
            effective_input: zeros4.clone(),
            output: zeros4.clone(),
            delta_out: zeros4.clone(),
        }
    }

    pub fn size(&self) -> usize {
        let (filters, in_channels, kernel_size, _) = self.kernel_shape;
        filters * in_channels * kernel_size * kernel_size + filters
    }

    pub fn forward(
        &mut self,
        params: &[f32],
        input: ArrayView4<f32>,
    ) -> Result<ArrayView4<'_, f32>> {
        let (kernel, bias) = self.view_params(params)?;

        let Self {
            kernel_shape,
            stride,
            padding,
            real_input_shape: ref mut real_input_dim,
            ref mut effective_input,
            ref mut output,
            ..
        } = *self;

        let (batch_size, channels, input_h, input_w) = input.dim();
        let (filters, _, kernel_size, _) = kernel_shape;

        *real_input_dim = input.dim();

        let out_h = (input_h + 2 * padding - kernel_size) / stride + 1;
        let out_w = (input_w + 2 * padding - kernel_size) / stride + 1;

        let effective_h = (out_h - 1) * stride + kernel_size;
        let effective_w = (out_w - 1) * stride + kernel_size;

        effective_input.reshape_inplace((batch_size, channels, effective_h, effective_w));
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
                // TODO: prealloc
                let col_image = Self::im2col(input_b, kernel_size, stride);
                // SAFETY: `kernel` has filters * in_channels * kernel_size^2 elements.
                let flat_kernel = kernel
                    .into_shape_with_order((filters, channels * kernel_size * kernel_size))
                    .unwrap();
                // SAFETY: `buf` was already reshaped to have enough elements.
                let mut flat_output = output_b
                    .view_mut()
                    .into_shape_with_order((filters, out_h * out_w))
                    .unwrap();

                linalg::general_mat_mul(1.0, &flat_kernel, &col_image, 0.0, &mut flat_output);
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

        let (filters, in_channels, kernel_size, _) = self.kernel_shape;
        let stride = self.stride;

        d_kernel.fill(0.);
        self.delta_out.reshape_inplace(self.real_input_shape);
        self.delta_out.fill(0.);

        let (_, _, delta_in_h, delta_in_w) = delta_in.dim();

        delta_in
            .axis_iter(Axis(0))
            .zip(self.effective_input.axis_iter(Axis(0)))
            .zip(self.delta_out.axis_iter_mut(Axis(0)))
            .for_each(|((delta_in_b, effective_input_b), mut delta_out_b)| {
                // delta out
                // SAFETY: `kernel` has filters * in_channels * kernel_size^2 elements.
                let flat_kernel = kernel
                    .into_shape_with_order((filters, in_channels * kernel_size * kernel_size))
                    .unwrap();
                // SAFETY: `delta_in_b` has filters * delta_h * delta_w elements.
                let flat_delta_in = delta_in_b
                    // TODO: delta_in_b.dim() por out_h y out_w
                    .into_shape_with_order((filters, delta_in_h * delta_in_w))
                    .unwrap();

                // TODO: prealloc
                let mut col_delta = Array2::zeros((
                    in_channels * kernel_size * kernel_size,
                    delta_in_h * delta_in_w,
                ));

                linalg::general_mat_mul(1.0, &flat_kernel.t(), &flat_delta_in, 0.0, &mut col_delta);
                Self::col2im_into(&mut delta_out_b, col_delta.view(), kernel_size, stride);

                // kernel grad
                // TODO: calcular una vez en forward
                let col_image = Self::im2col(effective_input_b, kernel_size, stride);

                // SAFETY: `d_kernel` has filters * in_channels * kernel_size^2 elements.
                let mut flat_d_kernel = d_kernel
                    .view_mut()
                    .into_shape_with_order((filters, in_channels * kernel_size * kernel_size))
                    .unwrap();
                linalg::general_mat_mul(
                    1.0,
                    &flat_delta_in,
                    &col_image.t(),
                    1.0,
                    &mut flat_d_kernel,
                );
            });

        let db_sum = delta_in
            .sum_axis(Axis(0))
            .sum_axis(Axis(1))
            .sum_axis(Axis(1));
        d_bias.assign(&db_sum);

        Ok(self.delta_out.view_mut())
    }

    // TODO: no es técnicamente un reshape porq estoy básicamente copiando píxeles q no habría si no
    /// Reshapes a three-dimension image tensor into a matrix with columns that match the window
    /// views that the kernel sees during the convolution pass.
    ///
    /// # Args
    /// * `image` - The **already padded** image tensor.
    /// * `kernel_size` - The size of the square kernel.
    /// * `stride` - The size of the kernel steps.
    ///
    /// # Returns
    /// The reshaped matrix from the image tensor.
    pub fn im2col(image: ArrayView3<f32>, kernel_size: usize, stride: usize) -> Array2<f32> {
        let (channels, image_h, image_w) = image.dim();

        let out_h = (image_h - kernel_size) / stride + 1;
        let out_w = (image_w - kernel_size) / stride + 1;

        // TODO: poner esto en metadata o algo
        let mut col_image = Array2::zeros((channels * kernel_size * kernel_size, out_h * out_w));

        for (row_idx, i) in (0..image_h - kernel_size + 1).step_by(stride).enumerate() {
            for (col_idx, j) in (0..image_w - kernel_size + 1).step_by(stride).enumerate() {
                let window = image.slice(s![.., i..i + kernel_size, j..j + kernel_size]);
                let mut col = col_image.column_mut(row_idx * out_w + col_idx);

                col.iter_mut().zip(window.iter()).for_each(|(c, w)| {
                    *c = *w;
                });
            }
        }

        col_image
    }

    fn col2im_into(
        im_delta: &mut ArrayViewMut3<f32>,
        col_delta: ArrayView2<f32>,
        kernel_size: usize,
        stride: usize,
    ) {
        let (_, image_h, image_w) = im_delta.dim();

        let out_w = (image_w - kernel_size) / stride + 1;

        // TODO: poner esto en metadata o algo
        // let mut image = Array3::<f32>::zeros(image_shape);

        for (row_idx, i) in (0..image_h - kernel_size + 1).step_by(stride).enumerate() {
            for (col_idx, j) in (0..image_w - kernel_size + 1).step_by(stride).enumerate() {
                let col = col_delta.column(row_idx * out_w + col_idx);
                let mut window = im_delta.slice_mut(s![.., i..i + kernel_size, j..j + kernel_size]);

                window.iter_mut().zip(col.iter()).for_each(|(w, c)| {
                    *w += *c;
                })
            }
        }
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
        let size = self.size();

        if params.len() != size {
            return Err(MlErr::size_mismatch("params", params.len(), size));
        }

        let (filters, in_channels, kernel_size, _) = self.kernel_shape;
        let weights_size = filters * in_channels * kernel_size * kernel_size;

        // SAFETY: The if condition above checks that the size of the
        //         parameters is exactly the size of the layer.
        let weights = ArrayView4::from_shape(self.kernel_shape, &params[..weights_size]).unwrap();
        let biases = ArrayView4::from_shape((1, filters, 1, 1), &params[weights_size..]).unwrap();

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
        let size = self.size();

        if grad.len() != size {
            return Err(MlErr::size_mismatch("grad", grad.len(), size));
        }

        let (filters, in_channels, kernel_size, _) = self.kernel_shape;
        let weights_size = filters * in_channels * kernel_size * kernel_size;

        // SAFETY: The if condition above checks that the size of the
        //         gradient is exactly the size of the layer.
        let (dw_raw, db_raw) = grad.split_at_mut(weights_size);

        let dw = ArrayViewMut4::from_shape(self.kernel_shape, dw_raw).unwrap();
        let db = ArrayViewMut1::from_shape(filters, db_raw).unwrap();

        Ok((dw, db))
    }
}
