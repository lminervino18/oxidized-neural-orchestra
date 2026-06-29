use std::cmp;

use ndarray::{linalg, prelude::*};
use ndarray_conv::{ConvExt, ConvMode, PaddingMode, ReverseKernel};

use crate::{MlErr, Result, arch::InplaceReshape};

#[derive(Clone, Debug)]
struct Convolver {
    // processor: Option<P>,
    // kernel_size: usize,
}

impl Convolver {
    fn new(_kernel_size: usize, _processor: Option<()>) -> Self {
        Self {}
    }

    /// Reshapes a three-dimension image tensor into a matrix with columns that match the window
    /// views that the kernel sees during the convolution pass.
    ///
    /// ## Args
    /// * `image` - The **already padded** image tensor.
    /// * `kernel_size` - The size of the square kernel.
    /// * `stride` - The size of the kernel steps.
    ///
    /// ## Returns
    /// The reshaped matrix from the image tensor.
    fn im2col(&self, image: ArrayView3<f32>, kernel_size: usize, stride: usize) -> Array2<f32> {
        let (channels, image_h, image_w) = image.dim();

        let out_h = (image_h - kernel_size) / stride + 1;
        let out_w = (image_w - kernel_size) / stride + 1;

        let mut col_image = Array2::zeros((channels * kernel_size * kernel_size, out_h * out_w));

        for (row_idx, i) in (0..image_h - kernel_size + 1).step_by(stride).enumerate() {
            for (col_idx, j) in (0..image_w - kernel_size + 1).step_by(stride).enumerate() {
                let window = image.slice(s![.., i..i + kernel_size, j..j + kernel_size]);
                let col = col_image.column_mut(row_idx * out_w + col_idx);

                // SAFETY: Both arrays have the same number of elements: kernel_size^2.
                col.into_shape_with_order(window.dim())
                    .unwrap()
                    .assign(&window);
            }
        }

        col_image
    }

    pub fn conv_into(
        &self,
        buf: &mut Array3<f32>,
        input: ArrayView3<f32>,
        kernel: ArrayView4<f32>,
        _reverse: bool, // wip
        stride: usize,
    ) -> Result<()> {
        let (channels, image_h, image_w) = input.dim();
        let (filters, in_channels, kernel_w, kernel_h) = kernel.dim();
        assert_eq!(channels, in_channels);
        assert_eq!(kernel_w, kernel_h);

        let kernel_size = kernel_w;
        let col_image = self.im2col(input, kernel_size, stride);

        // SAFETY: `kernel` has filters * in_channels * kernel_size^2 elements.
        let col_kernel = kernel
            .into_shape_with_order((filters, in_channels * kernel_size * kernel_size))
            .unwrap();

        let out_h = (image_h - kernel_size) / stride + 1;
        let out_w = (image_w - kernel_size) / stride + 1;

        buf.reshape_inplace((filters, out_h, out_w));
        // SAFETY: `buf` was just reshaped to have enough elements.
        let mut buf = buf
            .view_mut()
            .into_shape_with_order((filters, out_h * out_w))
            .unwrap();

        linalg::general_mat_mul(1.0, &col_kernel, &col_image, 0.0, &mut buf);
        Ok(())
    }

    fn conv2d(
        &mut self,
        input: ArrayView2<f32>,
        kernel: ArrayView2<f32>,
        reverse: bool,
        conv_mode: ConvMode<2>,
        padding_mode: PaddingMode<2, f32>,
    ) -> Result<Array2<f32>> {
        if reverse {
            Ok(input.conv(kernel.reverse(), conv_mode, padding_mode)?)
        } else {
            Ok(input.conv(kernel.no_reverse(), conv_mode, padding_mode)?)
        }
    }

    fn conv3d(
        &self,
        input: ArrayView3<f32>,
        kernel: ArrayView3<f32>,
        reverse: bool,
        conv_mode: ConvMode<3>,
        padding_mode: PaddingMode<3, f32>,
    ) -> Result<Array3<f32>> {
        if reverse {
            Ok(input.conv(kernel.reverse(), conv_mode, padding_mode)?)
        } else {
            Ok(input.conv(kernel.no_reverse(), conv_mode, padding_mode)?)
        }
    }
}

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
    dilated: Array4<f32>,

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
            dilated: zeros4,
            convolver,
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn forward(&mut self, params: &[f32], x: ArrayView4<f32>) -> Result<ArrayView4<'_, f32>> {
        let (k, b) = self.view_params(params)?;

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

        let (batch_size, _, input_height, input_width) = x.dim();

        *real_input_dim = (input_height, input_width);

        let output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
        let output_width = (input_width + 2 * padding - kernel_size) / stride + 1;

        let effective_height = (output_height - 1) * stride + kernel_size;
        let effective_width = (output_width - 1) * stride + kernel_size;

        effective_input.reshape_inplace((x.dim().0, x.dim().1, effective_height, effective_width));
        effective_input.fill(0.);

        // dropped elements could just be padding
        let copy_height = cmp::min(input_height, effective_height - padding);
        let copy_width = cmp::min(input_width, effective_width - padding);

        let mut effective_input_view = effective_input.slice_mut(s![
            ..,
            ..,
            padding..padding + copy_height,
            padding..padding + copy_width,
        ]);
        let input_view = &x.slice(s![.., .., ..copy_height, ..copy_width]);
        effective_input_view.assign(input_view);

        output.reshape_inplace((batch_size, filters, output_height, output_width));

        effective_input
            .axis_iter(Axis(0))
            .zip(output.axis_iter_mut(Axis(0)))
            .try_for_each(|(input_b, mut output_b)| -> Result<()> {
                for f in 0..filters {
                    let kernel_f = k.index_axis(Axis(0), f);

                    let res_3d = convolver.conv3d(
                        input_b,
                        kernel_f,
                        false,
                        ConvMode::Custom {
                            padding: [0; 3],
                            strides: [1, stride, stride],
                        },
                        PaddingMode::Zeros,
                    )?;
                    let res_2d = res_3d.index_axis(Axis(0), 0);

                    output_b.slice_mut(s![f, .., ..]).assign(&res_2d);
                }

                Ok(())
            })?;

        *output += &b;

        Ok(output.view())
    }

    pub fn backward(
        &mut self,
        params: &[f32],
        grad: &mut [f32],
        d_in: ArrayViewMut4<f32>,
    ) -> Result<ArrayViewMut4<'_, f32>> {
        let (mut dk, mut db) = self.view_grad(grad)?;
        let (k, _) = self.view_params(params)?;

        self.dilate(d_in.view());

        let Self {
            filters,
            in_channels,
            padding,
            real_input_dim,
            ref effective_input,
            ref mut delta_out,
            ref mut dilated,
            ref mut convolver,
            ..
        } = *self;

        dk.fill(0.);
        delta_out.reshape_inplace((
            effective_input.dim().0,
            effective_input.dim().1,
            real_input_dim.0,
            real_input_dim.1,
        ));
        delta_out.fill(0.);

        dilated
            .axis_iter(Axis(0))
            .zip(effective_input.axis_iter(Axis(0)))
            .zip(delta_out.axis_iter_mut(Axis(0)))
            .try_for_each(
                |((dilated_b, effective_input_b), mut delta_out_b)| -> Result<()> {
                    for f_idx in 0..filters {
                        let dilated_bf = dilated_b.slice(s![f_idx, .., ..]);

                        for c_idx in 0..in_channels {
                            // kernel
                            let effective_input_bc = effective_input_b.slice(s![c_idx, .., ..]);

                            let dk_step = convolver.conv2d(
                                effective_input_bc,
                                dilated_bf,
                                false,
                                ConvMode::Valid,
                                PaddingMode::Zeros,
                            )?;

                            let mut dk_view = dk.slice_mut(s![f_idx, c_idx, .., ..]);
                            dk_view += &dk_step;

                            // delta
                            let k_fc = k.slice(s![f_idx, c_idx, .., ..]);

                            let copy_height =
                                cmp::min(real_input_dim.0, effective_input.dim().2 - padding);
                            let copy_width =
                                cmp::min(real_input_dim.1, effective_input.dim().3 - padding);

                            let effective_delta_step = convolver.conv2d(
                                dilated_bf,
                                k_fc,
                                true,
                                ConvMode::Full,
                                PaddingMode::Zeros,
                            )?;
                            let delta_step = effective_delta_step.slice(s![
                                padding..padding + copy_height,
                                padding..padding + copy_width
                            ]);

                            let mut delta_view =
                                delta_out_b.slice_mut(s![c_idx, ..copy_height, ..copy_width]);
                            delta_view += &delta_step;
                        }
                    }

                    Ok(())
                },
            )?;

        let db_sum = d_in.sum_axis(Axis(0)).sum_axis(Axis(1)).sum_axis(Axis(1));
        db.assign(&db_sum);

        Ok(delta_out.view_mut())
    }

    /// Performs inward dilation to a input delta and saves the result into the delta metadata
    /// array.
    ///
    /// ## Args
    /// * `delta` - The input delta to dilate and pad.
    fn dilate(&mut self, delta: ArrayView4<f32>) {
        let Self {
            stride,
            ref mut dilated,
            ..
        } = *self;

        let inward_padding = stride - 1;
        let (delta_filters, delta_in_channels, delta_width, delta_height) = delta.dim();
        let dilated_width = delta_width + (delta_width - 1) * inward_padding;
        let dilated_height = delta_height + (delta_height - 1) * inward_padding;

        let dilated_dim = (
            delta_filters,
            delta_in_channels,
            dilated_height,
            dilated_width,
        );

        dilated.reshape_inplace(dilated_dim);
        // NOTE: this might not be needed as the assigned delta overwrites the past one if
        // dimensions match.
        dilated.fill(0.);
        dilated
            .slice_mut(s![.., ..,
                ..dilated_height; stride,
                ..dilated_width; stride])
            .assign(&delta);
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

#[cfg(test)]
mod tests {
    use std::env;

    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_conv2d00_forward_backward_consistency() {
        unsafe {
            std::env::set_var("RUST_BACKTRACE", "1");
        }

        let filters = 1;
        let in_channels = 1;
        let kernel_size = 2;
        let stride = 2;
        let padding = 0;
        let mut layer = Conv2d::new(filters, in_channels, kernel_size, stride, padding);

        let input = array![[[
            [1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.]
        ]]];
        let params: Vec<_> = (0..layer.size()).map(|i| i as f32 / 10.).collect();
        let mut grads = vec![0.; layer.size()];

        let output = layer.forward(&params, input.view()).unwrap();
        assert_eq!(output.dim(), (1, 1, 2, 2));

        let d_out = Array4::from_elem((1, 1, 2, 2), 1.);
        layer
            .backward(&params, &mut grads, d_out.view().to_owned().view_mut())
            .unwrap();

        assert!((grads[4] - 4.).abs() < 1e-5);
    }

    #[test]
    fn test_conv2d01_dilate() {
        let filters = 1;
        let in_channels = 1;
        let kernel_size = 2;
        let stride = 2;
        let padding = 0;
        let mut conv = Conv2d::new(filters, in_channels, kernel_size, stride, padding);

        let delta: Array4<f32> = array![[[
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.],
            [13., 14., 15., 16.]
        ]]];

        let expected = array![[[
            [1., 0., 2., 0., 3., 0., 4.],
            [0., 0., 0., 0., 0., 0., 0.],
            [5., 0., 6., 0., 7., 0., 8.],
            [0., 0., 0., 0., 0., 0., 0.],
            [9., 0., 10., 0., 11., 0., 12.],
            [0., 0., 0., 0., 0., 0., 0.],
            [13., 0., 14., 0., 15., 0., 16.]
        ]]];

        conv.dilate(delta.view());

        assert_eq!(conv.dilated, expected);
    }

    #[test]
    fn test_conv2d02_dilate_with_no_stride_does_not_change_delta() {
        let filters = 1;
        let in_channels = 1;
        let kernel_size = 2;
        let stride = 1;
        let padding = 0;
        let mut conv = Conv2d::new(filters, in_channels, kernel_size, stride, padding);

        let delta: Array4<f32> = array![[[
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.],
            [13., 14., 15., 16.]
        ]]];

        conv.dilate(delta.view());

        assert_eq!(conv.dilated, delta);
    }

    fn test_conv2d_forward(
        filters: usize,
        in_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        params: &[f32],
        input: &Array4<f32>,
        expected: &Array4<f32>,
    ) {
        let mut conv = Conv2d::new(filters, in_channels, kernel_size, stride, padding);
        let output = conv.forward(params, input.view()).unwrap();
        // println!("output:\n{:#}", output);
        assert_abs_diff_eq!(output, expected, epsilon = 1e-4);
    }

    #[test]
    fn test_conv2d03_00forward_one_filter_one_in_channel_kernel_size2_one_stride_no_padding() {
        let params: [f32; 5] = [
            1., 2., 3., 4., // filter
            5., // bias
        ];
        let input: Array4<f32> = array![[[[1., 2., 3.,], [4., 5., 6.], [7., 8., 9.]]]];
        let expected = array![
            // first sample
            [
                // first input channel
                [[42., 52.], [72., 82.]]
            ]
        ];
        test_conv2d_forward(1, 1, 2, 1, 0, &params, &input, &expected);
    }

    #[test]
    fn test_conv2d04_01forward_one_filter_one_in_channel_kernel_size2_one_stride_padding1() {
        let params: [f32; 5] = [
            1., 2., 3., 4., // filter
            5., // bias
        ];
        let input: Array4<f32> = array![
            // first sample
            [
                // first input channel
                [[1., 2., 3.,], [4., 5., 6.], [7., 8., 9.]]
            ]
        ];
        let expected = array![
            // first convoluted sample
            [
                // first convoluted in channel
                [
                    [9., 16., 23., 14.],
                    [23., 42., 52., 26.],
                    [41., 72., 82., 38.],
                    [19., 28., 31., 14.]
                ]
            ]
        ];
        test_conv2d_forward(1, 1, 2, 1, 1, &params, &input, &expected);
    }

    #[test]
    fn test_conv2d05_forward_filters1_in_channels1_kernel_size2_one_stride_padding1_batch_size2() {
        let params: [f32; 5] = [
            1., 2., 3., 4., // filter
            5., // bias
        ];
        let input: Array4<f32> = array![
            [[[1., 2., 3.,], [4., 5., 6.], [7., 8., 9.]]],
            [[[1., 2., 3.,], [4., 5., 6.], [7., 8., 9.]]]
        ];
        let expected = array![
            [[
                [9., 16., 23., 14.],
                [23., 42., 52., 26.],
                [41., 72., 82., 38.],
                [19., 28., 31., 14.]
            ]],
            [[
                [9., 16., 23., 14.],
                [23., 42., 52., 26.],
                [41., 72., 82., 38.],
                [19., 28., 31., 14.]
            ]]
        ];
        test_conv2d_forward(1, 1, 2, 1, 1, &params, &input, &expected);
    }

    #[test]
    fn test_conv2d06_forward_filters2_in_channels1_kernel_size2_one_stride_padding1_batch_size2() {
        let params: [f32; 10] = [1., 2., 3., 4., 1., 2., 3., 4., 5., 5.];
        let input: Array4<f32> = array![
            [[[1., 2., 3.,], [4., 5., 6.], [7., 8., 9.]]],
            [[[1., 2., 3.,], [4., 5., 6.], [7., 8., 9.]]]
        ];
        let expected = array![
            [
                [
                    [9., 16., 23., 14.],
                    [23., 42., 52., 26.],
                    [41., 72., 82., 38.],
                    [19., 28., 31., 14.]
                ],
                [
                    [9., 16., 23., 14.],
                    [23., 42., 52., 26.],
                    [41., 72., 82., 38.],
                    [19., 28., 31., 14.]
                ]
            ],
            [
                [
                    [9., 16., 23., 14.],
                    [23., 42., 52., 26.],
                    [41., 72., 82., 38.],
                    [19., 28., 31., 14.]
                ],
                [
                    [9., 16., 23., 14.],
                    [23., 42., 52., 26.],
                    [41., 72., 82., 38.],
                    [19., 28., 31., 14.]
                ]
            ]
        ];
        test_conv2d_forward(2, 1, 2, 1, 1, &params, &input, &expected);
    }

    #[test]
    fn test_conv2d07_forward_filters2_in_channels2_kernel_size2_one_stride_padding1_batch_size2() {
        let params: [f32; 18] = [
            1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4., 5., 5.,
        ];
        let input: Array4<f32> = array![
            [
                [[1., 2., 3.,], [4., 5., 6.], [7., 8., 9.]],
                [[1., 2., 3.,], [4., 5., 6.], [7., 8., 9.]]
            ],
            [
                [[1., 2., 3.,], [4., 5., 6.], [7., 8., 9.]],
                [[1., 2., 3.,], [4., 5., 6.], [7., 8., 9.]]
            ]
        ];
        let expected = array![
            [
                [
                    [13.0, 27.0, 41.0, 23.0],
                    [41.0, 79.0, 99.0, 47.0],
                    [77.0, 139.0, 159.0, 71.0],
                    [33.0, 51.0, 57.0, 23.0]
                ],
                [
                    [13.0, 27.0, 41.0, 23.0],
                    [41.0, 79.0, 99.0, 47.0],
                    [77.0, 139.0, 159.0, 71.0],
                    [33.0, 51.0, 57.0, 23.0]
                ]
            ],
            [
                [
                    [13.0, 27.0, 41.0, 23.0],
                    [41.0, 79.0, 99.0, 47.0],
                    [77.0, 139.0, 159.0, 71.0],
                    [33.0, 51.0, 57.0, 23.0]
                ],
                [
                    [13.0, 27.0, 41.0, 23.0],
                    [41.0, 79.0, 99.0, 47.0],
                    [77.0, 139.0, 159.0, 71.0],
                    [33.0, 51.0, 57.0, 23.0]
                ]
            ]
        ];
        test_conv2d_forward(2, 2, 2, 1, 1, &params, &input, &expected);
    }

    fn test_conv2d_backward(
        conv: &mut Conv2d,
        params: &[f32],
        input: &Array4<f32>,
        delta_in: &mut Array4<f32>,
        expected_delta_out: &Array4<f32>,
        expected_grad: &[f32],
    ) {
        let mut grad = vec![0.; params.len()];
        let _ = conv.forward(params, input.view()).unwrap();
        let delta_out = conv
            .backward(params, &mut grad, delta_in.view_mut())
            .unwrap();
        // println!("delta_out:\n{:#}", delta_out);
        assert_eq!(delta_out, expected_delta_out);
        // println!("grad:\n{:#?}", grad);
        assert_eq!(grad, expected_grad);
    }

    #[test]
    fn test_conv2d08_backward_filters1_in_channels1_kernel_size2_stride2_padding0() {
        let mut conv = Conv2d::new(1, 1, 2, 2, 0);
        let params: [f32; 5] = [1., 2., 3., 4., 5.];
        let input: Array4<f32> = array![[[
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.],
            [13., 14., 15., 16.]
        ]]];
        let mut delta_in: Array4<f32> = array![[[[17., 18.], [19., 20.]]]];
        let expected_delta_out = array![[[
            [17., 34., 18., 36.],
            [51., 68., 54., 72.],
            [19., 38., 20., 40.],
            [57., 76., 60., 80.]
        ]]];
        let expected_grad = [462., 536., 758., 832., 74.];
        test_conv2d_backward(
            &mut conv,
            &params,
            &input,
            &mut delta_in,
            &expected_delta_out,
            &expected_grad,
        );
    }

    #[test]
    fn test_conv2d09_backward_should_return_same_output_if_ran_twice() {
        let mut conv = Conv2d::new(1, 1, 2, 2, 0);
        let params: [f32; 5] = [1., 2., 3., 4., 5.];
        let input: Array4<f32> = array![[[
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.],
            [13., 14., 15., 16.]
        ]]];
        let mut delta_in: Array4<f32> = array![[[[17., 18.], [19., 20.]]]];
        let expected_delta_out = array![[[
            [17., 34., 18., 36.],
            [51., 68., 54., 72.],
            [19., 38., 20., 40.],
            [57., 76., 60., 80.]
        ]]];
        let expected_grad = [462., 536., 758., 832., 74.];
        test_conv2d_backward(
            &mut conv,
            &params,
            &input,
            &mut delta_in,
            &expected_delta_out,
            &expected_grad,
        );
        test_conv2d_backward(
            &mut conv,
            &params,
            &input,
            &mut delta_in,
            &expected_delta_out,
            &expected_grad,
        );
    }

    // The forward convolution will drop values that don't fit.
    // In this example:
    // * input (w/ padding): 10x10
    // * filter: 3x3
    // With a stride of 2, the filter fits 4 times before there is one element missing in the
    // width dimension. We could add more padding or just drop the element.
    #[test]
    fn test_conv2d10_forward_with_input_kernel_convolution_mismatch() {
        let input_height = 8;
        let input_width = 8;
        let kernel_size = 3;
        let params = vec![0.; kernel_size * kernel_size + 1];
        let input = Array4::from_elem((1, 1, input_height, input_width), 0.);
        let expected = Array4::from_elem((1, 1, 4, 4), 0.);
        test_conv2d_forward(1, 1, kernel_size, 2, 1, &params, &input, &expected);
    }

    // So, what happens with the backward pass convolution?
    // Well, first the convolution between the input and the dilated upstream delta should match
    // the dimensionality of the kernel gradient, which is just the dimensionality of the kernel.
    // In this example that's 3x3.
    // The problem is that the actual convolution between the 10x10 padded input and the dilated
    // upstream 7x7 delta outputs a 4x4 matrix (without having into account the extra higher
    // dimensions).
    // The solution is to compute the convolution a *effective* input and the dilated upstread
    // delta. With effective input I mean the input that was actually used for the convolution in
    // the forward pass, so that means dropping the right-most elements that the kernel did not
    // touch there.
    #[test]
    fn test_conv2d11_backward_with_input_kernel_convolution_mismatch() {
        unsafe { env::set_var("RUST_BACKTRACE", "1") };

        let input_height = 8;
        let input_width = 8;
        let kernel_size = 3;
        let params = vec![0.; kernel_size * kernel_size + 1];
        let input = Array4::from_elem((1, 1, input_height, input_width), 0.);
        // the dimensionality of the upstream delta is the same that as the output
        let output_height = 4;
        let output_width = 4;
        let mut delta_in = Array4::from_elem((1, 1, output_height, output_width), 0.);
        let mut conv = Conv2d::new(1, 1, kernel_size, 2, 1);

        let expected_delta_out = Array4::from_elem((1, 1, input_height, input_width), 0.);
        let expected_grad = vec![0.; kernel_size * kernel_size + 1];

        test_conv2d_backward(
            &mut conv,
            &params,
            &input,
            &mut delta_in,
            &expected_delta_out,
            &expected_grad,
        );
    }
}
