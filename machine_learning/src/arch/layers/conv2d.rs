use ndarray::prelude::*;
use ndarray_conv::{ConvExt, ConvMode, PaddingMode, ReverseKernel};

use crate::{MlErr, Result, arch::InplaceReshape};

#[derive(Clone)]
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
    conv_mode: ConvMode<3>,

    // Backward metadata
    delta: Array4<f32>,
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
        let conv_mode = ConvMode::Custom {
            padding: [0, padding, padding],
            strides: [1, stride, stride],
        };

        Self {
            filters,
            in_channels,
            kernel_size,
            stride,
            padding,
            kernels_size,
            kernels_dim,
            size,
            conv_mode,
            input: zeros4.clone(),
            output: zeros4.clone(),
            delta: zeros4.clone(),
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
            conv_mode,
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
                let res_3d = input_b.conv(kernel_f.no_reverse(), conv_mode, PaddingMode::Zeros)?;
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
        d: ArrayViewMut4<f32>,
    ) -> Result<ArrayViewMut4<'_, f32>> {
        let (mut dw, mut db) = self.view_grad(grad)?;
        let (k, _) = self.view_params(params)?;

        self.dilate_and_pad(d.view());

        let Self {
            kernel_size,
            padding,
            ref input,
            ref mut delta,
            ref mut dilated,
            ..
        } = *self;

        let outward_padding = kernel_size - padding - 1;
        let (_, _, dilated_width, dilated_height) = dilated.dim();

        let unpadded_dilated = dilated.slice(s![
            ..,
            ..,
            outward_padding..dilated_width - outward_padding,
            outward_padding..dilated_height - outward_padding
        ]);

        let (batch_size, in_channels, _, _) = input.dim();
        let (filters, _, _, _) = k.dim();

        delta.reshape_inplace(input.dim());

        for b in 0..batch_size {
            for f in 0..filters {
                for c in 0..in_channels {
                    // kernel
                    let x_bc = input.slice(s![b, c, .., ..]);
                    let ud_bf = unpadded_dilated.slice(s![b, f, .., ..]);

                    let step =
                        x_bc.conv(ud_bf.no_reverse(), ConvMode::Valid, PaddingMode::Zeros)?;

                    let mut dw_view = dw.slice_mut(s![f, c, .., ..]);
                    dw_view += &step;

                    // bias
                    let d_bf = dilated.slice(s![b, f, .., ..]);
                    let w_fc = k.slice(s![f, c, .., ..]);

                    let step = d_bf
                        .conv(&w_fc, ConvMode::Valid, PaddingMode::Zeros)
                        .unwrap();

                    let mut delta_view = delta.slice_mut(s![b, c, .., ..]);
                    delta_view += &step;
                }
            }
        }

        let db_sum = d.sum_axis(Axis(0)).sum_axis(Axis(1)).sum_axis(Axis(1));
        db.assign(&db_sum);

        Ok(delta.view_mut())
    }

    /// Performs inward and outward (padding) dilations to a input delta and saves the result into
    /// the delta metadata array.
    ///
    /// ## Args
    /// * `delta` - The input delta to dilate and pad.
    fn dilate_and_pad(&mut self, delta: ArrayView4<f32>) {
        let Self {
            stride,
            kernel_size,
            padding,
            ref mut dilated,
            ..
        } = *self;

        let inward_padding = stride - 1;
        let outward_padding = kernel_size - padding - 1;
        let delta_filters = delta.dim().0;
        let delta_in_channels = delta.dim().1;
        let delta_width = delta.dim().3;
        let delta_height = delta.dim().2;

        let dilated_width = delta_width + (delta_width - 1) * inward_padding + outward_padding * 2;
        let dilated_height =
            delta_height + (delta_height - 1) * inward_padding + outward_padding * 2;

        let dilated_dim = (
            delta_filters,
            delta_in_channels,
            dilated_height,
            dilated_width,
        );

        dilated.reshape_inplace(dilated_dim);
        dilated
            .slice_mut(s![.., ..,
                outward_padding..dilated_height - outward_padding; stride,
                outward_padding..dilated_width - outward_padding; stride])
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
    fn test_conv2d_forward_backward_consistency() {
        let filters = 1;
        let in_channels = 1;
        let kernel_size = 2;
        let stride = 2;
        let padding = 0;
        let mut layer = Conv2d::new(filters, in_channels, kernel_size, stride, padding);

        let input = Array4::from_elem((1, 1, 4, 4), 1.);
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
    fn test_conv2d_forward() {
        let filters = 1;
        let in_channels = 1;
        let kernel_size = 2;
        let stride = 2;
        let padding = 0;
        let mut conv = Conv2d::new(filters, in_channels, kernel_size, stride, padding);

        let input: Array4<f32> = array![[[
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.],
            [13., 14., 15., 16.]
        ]]];

        let params: [f32; 5] = [1., 2., 3., 4., 5.];
        let expected_output = array![[[[49., 69.], [129., 149.]]]];

        let output = conv.forward(&params[..], input.view()).unwrap();

        assert_eq!(output, expected_output);
    }

    #[test]
    fn test_dilate_and_pad() {
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
            [0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 0., 2., 0., 3., 0., 4., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 5., 0., 6., 0., 7., 0., 8., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 9., 0., 10., 0., 11., 0., 12., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 13., 0., 14., 0., 15., 0., 16., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0.],
        ]]];

        conv.dilate_and_pad(delta.view());
        let got = conv.dilated;

        assert_eq!(got, expected);
    }

    #[test]
    fn test_conv2d_backward() {
        let filters = 1;
        let in_channels = 1;
        let kernel_size = 2;
        let stride = 2;
        let padding = 0;
        let mut conv = Conv2d::new(filters, in_channels, kernel_size, stride, padding);

        let input: Array4<f32> = array![[[
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.],
            [13., 14., 15., 16.]
        ]]];

        let params: [f32; 5] = [1., 2., 3., 4., 5.];
        let mut grad: [f32; 5] = [0., 0., 0., 0., 0.];
        let _ = conv.forward(&params, input.view()).unwrap();

        let mut delta_in: Array4<f32> = array![[[[17., 18.], [19., 20.]]]];

        let expected_delta_out = array![[[
            [17., 34., 18., 36.],
            [51., 68., 54., 72.],
            [19., 38., 20., 40.],
            [57., 76., 60., 80.]
        ]]];

        let expected_grad = [462., 536., 758., 832., 74.];

        let delta_out = conv
            .backward(&params, &mut grad, delta_in.view_mut())
            .unwrap();

        assert_eq!(delta_out, expected_delta_out);
        assert_eq!(grad, expected_grad);
    }
}
