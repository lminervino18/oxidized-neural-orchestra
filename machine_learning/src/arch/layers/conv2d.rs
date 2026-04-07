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
    conv_mode: ConvMode<4>,

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
            padding: [0, 0, padding, padding],
            strides: [1, 1, stride, stride],
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
        self.input.reshape_inplace(x.raw_dim());
        self.input.assign(&x);

        let (k, b) = self.view_params(params)?;

        // estos errores no los vamos a estar manejando, quisiera que fuera más ergonómico
        // sacarlos, armar uno de los que teníamos acá es medio paja para hacerlo en cada forward,
        // si vamos a estar esperando errores frecuentes bueno este to_string evidentemente está
        // mal, me gustaría ver qué podemos hacer para que quede cool
        let mut output = x
            .conv(k.no_reverse(), self.conv_mode, PaddingMode::Zeros)
            .map_err(|e| MlErr::MatrixError {
                error: e.to_string(),
            })?;
        output += &b.into_dyn();

        self.output = output;

        Ok(self.output.view())
    }

    pub fn backward(
        &mut self,
        params: &[f32],
        grad: &mut [f32],
        d: ArrayViewMut4<f32>,
    ) -> Result<ArrayViewMut4<'_, f32>> {
        let (mut dw, mut db) = self.view_grad(grad)?;
        let (w, _) = self.view_params(params)?;

        self.dilate_and_pad(d.view())?;

        let Self {
            padding,
            kernel_size,
            ref input,
            ref mut delta,
            ref mut dilated,
            ..
        } = *self;

        let outward_padding = kernel_size - padding - 1;
        let unpadded_dilated = dilated.slice(s![
            ..,
            ..,
            outward_padding..dilated.dim().2 - outward_padding,
            outward_padding..dilated.dim().3 - outward_padding
        ]);

        let dw_conv = input
            .conv(
                unpadded_dilated.no_reverse(),
                ConvMode::Valid,
                PaddingMode::Zeros,
            )
            .unwrap();

        dw.assign(&dw_conv);

        let delta_new = dilated
            .conv(&w, ConvMode::Valid, PaddingMode::Zeros)
            .unwrap();

        delta.reshape_inplace(delta_new.dim());
        delta.assign(&delta_new);

        let db_sum = d.sum_axis(Axis(0)).sum_axis(Axis(1)).sum_axis(Axis(1));
        db.assign(&db_sum);

        Ok(delta.view_mut())
    }

    /// Performs inward and outward (padding) dilations to a input delta.
    ///
    /// ## Args
    /// * `delta` - The input delta to dilate and pad.
    fn dilate_and_pad(&mut self, delta: ArrayView4<f32>) -> Result<()> {
        let Self {
            stride,
            kernel_size,
            padding,
            ref mut dilated,
            ..
        } = *self;

        let inward_padding = stride - 1;
        let outward_padding = kernel_size - padding - 1;

        let dilated_height =
            delta.dim().2 + (delta.dim().2 - 1) * inward_padding + outward_padding * 2;
        let dilated_width =
            delta.dim().3 + (delta.dim().3 - 1) * inward_padding + outward_padding * 2;

        let dilated_dim = (delta.dim().0, delta.dim().1, dilated_height, dilated_width);

        dilated.reshape_inplace(dilated_dim);
        dilated
            .slice_mut(s![.., ..,
                outward_padding..dilated_height - outward_padding; stride,
                outward_padding..dilated_width - outward_padding; stride])
            .assign(&delta);

        Ok(())
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
    ) -> Result<(ArrayView4<'a, f32>, ArrayView1<'a, f32>)> {
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
        let biases = ArrayView1::from_shape(filters, &params[kernels_size..]).unwrap();

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

        conv.dilate_and_pad(delta.view()).unwrap();
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
