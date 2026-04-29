use ndarray::prelude::*;
use ndarray_conv::{ConvExt, ConvMode, PaddingMode, ReverseKernel};

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
    input: Array4<f32>,
    output: Array4<f32>,

    // Backward metadata
    delta_out: Array4<f32>,
    dilated: Array4<f32>,
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

        let zeros4 = Array4::zeros((1, 1, 1, 1));

        Self {
            filters,
            in_channels,
            kernel_size,
            stride,
            padding,
            kernels_size,
            kernels_dim,
            size,
            input: zeros4.clone(),
            output: zeros4.clone(),
            delta_out: zeros4.clone(),
            dilated: zeros4,
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
            ref mut input,
            ref mut output,
            ..
        } = *self;

        input.reshape_inplace(x.raw_dim());
        input.assign(&x);

        let (batch_size, _, input_height, input_width) = x.dim();

        let output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
        let output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
        output.reshape_inplace((batch_size, filters, output_height, output_width));

        for b in 0..batch_size {
            let input_b = x.index_axis(Axis(0), b);

            for f in 0..filters {
                let kernel_f = k.index_axis(Axis(0), f);
                let res_3d = input_b.conv(
                    kernel_f.no_reverse(),
                    ConvMode::Custom {
                        padding: [0, padding, padding],
                        strides: [1, stride, stride],
                    },
                    PaddingMode::Zeros,
                )?;
                let res_2d = res_3d.index_axis(Axis(0), 0);

                output.slice_mut(s![b, f, .., ..]).assign(&res_2d);
            }
        }

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
            kernel_size,
            padding,
            ref input,
            ref mut delta_out,
            ref mut dilated,
            ..
        } = *self;

        let batch_size = input.dim().0;

        delta_out.reshape_inplace(input.dim());
        delta_out.fill(0.);

        for b_idx in 0..batch_size {
            for f_idx in 0..filters {
                let dilated_bf = dilated.slice(s![b_idx, f_idx, .., ..]);

                for c_idx in 0..in_channels {
                    // kernel
                    let input_bc = input.slice(s![b_idx, c_idx, .., ..]);

                    let step = input_bc.conv(
                        dilated_bf.no_reverse(),
                        ConvMode::Custom {
                            padding: [padding; 2],
                            strides: [1; 2],
                        },
                        PaddingMode::Zeros,
                    )?;

                    let mut dk_view = dk.slice_mut(s![f_idx, c_idx, .., ..]);
                    dk_view += &step;

                    // delta
                    let k_fc = k.slice(s![f_idx, c_idx, .., ..]);

                    let step = dilated_bf.conv(
                        &k_fc,
                        ConvMode::Custom {
                            padding: [kernel_size - padding - 1; 2],
                            strides: [1; 2],
                        },
                        PaddingMode::Zeros,
                    )?;

                    let mut delta_view = delta_out.slice_mut(s![b_idx, c_idx, .., ..]);
                    delta_view += &step;
                }
            }
        }

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
        // dimensions match. I leave it commented out as it's pretty expensive to fill up the whole
        // dilated tensor with zeros.
        // dilated.fill(0.);
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
            return Err(MlErr::SizeMismatch {
                what: "params",
                got: params.len(),
                expected: size,
            });
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
            in_channels,
            kernel_size,
            kernels_size,
            size,
            ..
        } = *self;

        if grad.len() != size {
            return Err(MlErr::SizeMismatch {
                what: "grad",
                got: grad.len(),
                expected: self.size,
            });
        }

        // SAFETY: The if condition above checks that the size of the
        //         gradient is exactly the size of the layer.
        let (dw_raw, db_raw) = grad.split_at_mut(kernels_size);

        let kernels_dim = (filters, in_channels, kernel_size, kernel_size);
        let dw = ArrayViewMut4::from_shape(kernels_dim, dw_raw).unwrap();
        let db = ArrayViewMut1::from_shape(filters, db_raw).unwrap();

        Ok((dw, db))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[allow(clippy::too_many_arguments)]
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
        println!("output:\n{:#}", output);
        assert_eq!(output, expected);
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
        assert_eq!(delta_out, expected_delta_out);
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
}
