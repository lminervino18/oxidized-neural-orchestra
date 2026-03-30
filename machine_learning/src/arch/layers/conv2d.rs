use ndarray::prelude::*;
use ndarray_conv::{ConvExt, ConvMode, PaddingMode, ReverseKernel};

use crate::{MlErr, Result, arch::InplaceReshape};

#[derive(Clone)]
pub struct Conv2d {
    kernel_dim: (usize, usize, usize, usize),
    stride: usize,
    padding: usize,
    size: usize,
    w_size: usize,

    // Input metadata
    input: Array4<f32>,
    output: Array4<f32>,

    // Output metadata
    delta: Array4<f32>,
    dilated: Array4<f32>,
}

impl Conv2d {
    pub fn new(
        filters: usize,
        in_channels: usize,
        kernel_size: (usize, usize),
        stride: usize,
        padding: usize,
    ) -> Self {
        let (kh, kw) = kernel_size;
        let w_size = filters * in_channels * kh * kw;
        let size = w_size + filters;
        let zeros4 = Array4::zeros((1, 1, 1, 1));

        Self {
            kernel_dim: (filters, in_channels, kh, kw),
            stride,
            padding,
            size,
            w_size,
            input: zeros4.clone(),
            output: zeros4.clone(),
            delta: zeros4.clone(),
            dilated: zeros4.clone(),
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn forward(&mut self, params: &[f32], x: ArrayView4<f32>) -> Result<ArrayView4<'_, f32>> {
        let Self {
            stride, padding, ..
        } = *self;

        self.input.reshape_inplace(x.raw_dim());
        self.input.assign(&x);

        let (k, b) = self.view_params(params)?;
        let conv_mode = ConvMode::Custom {
            padding: [0, 0, padding, padding],
            strides: [1, 1, stride, stride],
        };

        let mut output = x.conv(&k, conv_mode, PaddingMode::Zeros).unwrap();
        let b_reshaped = b
            .view()
            .insert_axis(Axis(0))
            .insert_axis(Axis(2))
            .insert_axis(Axis(3));

        output += &b_reshaped;
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

        self.dilate_and_pad(d.view()).unwrap();

        let kernel_size = self.kernel_dim.2; // I'm assuming square kernel matrices for now
        let outward_padding = kernel_size - self.padding - 1;
        let unpadded_dilated = &self.dilated.slice(s![
            ..,
            ..,
            outward_padding..self.dilated.dim().2 - outward_padding,
            outward_padding..self.dilated.dim().3 - outward_padding
        ]);

        let dw_conv = self
            .input
            .conv(unpadded_dilated, ConvMode::Valid, PaddingMode::Zeros)
            .unwrap();

        dw.assign(&dw_conv);
        // dw.assign(&dw_conv.slice(s![.., .., ..self.dilated.dim().2, ..self.dilated.dim().3,]));

        let delta = self
            .dilated
            .conv(w.no_reverse(), ConvMode::Valid, PaddingMode::Zeros)
            .unwrap();

        self.delta.reshape_inplace(delta.dim());
        self.delta.assign(&delta);

        let db_sum = d.sum_axis(Axis(0)).sum_axis(Axis(1)).sum_axis(Axis(1));
        db.assign(&db_sum);

        Ok(self.delta.view_mut())
    }

    /// Performs inward and outward (padding) dilations to a input delta.
    ///
    /// This method assumes:
    /// * That a forward pass has been performed on the convolutional layer.
    //  TODO: make it so that this first assumption is not necessary
    /// * (For now) That the kernel is a square matrix.
    ///
    /// ## Args
    /// * `delta` - The input delta to dilate and pad.
    ///
    /// ## Panics
    /// Panics if the current `dilated` buffer doesn't have a corresponding
    /// shape.
    fn dilate_and_pad(&mut self, delta: ArrayView4<f32>) -> Result<()> {
        let inward_padding = self.stride - 1;
        let kernel_size = self.kernel_dim.2; // I'm assuming square kernel matrices for now
        let outward_padding = kernel_size - self.padding - 1;

        let dilated_dim = (
            delta.dim().0,
            delta.dim().1,
            delta.dim().2 + (delta.dim().2 - 1) * inward_padding + outward_padding * 2,
            delta.dim().3 + (delta.dim().3 - 1) * inward_padding + outward_padding * 2,
        );

        self.dilated.reshape_inplace(dilated_dim);
        self.dilated
            .slice_mut(s![.., ..,
                outward_padding..dilated_dim.2 - outward_padding; self.stride,
                outward_padding..dilated_dim.3 - outward_padding; self.stride])
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
        if params.len() != self.size {
            return Err(MlErr::SizeMismatch {
                what: "params",
                got: params.len(),
                expected: self.size,
            });
        }

        let (f, ..) = self.kernel_dim;
        let w_size = self.w_size;

        // SAFETY: The if condition above checks that the size of the
        //         parameters is exactly the size of the layer.
        let weights = ArrayView4::from_shape(self.kernel_dim, &params[..w_size]).unwrap();
        let biases = ArrayView1::from_shape(f, &params[w_size..]).unwrap();

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
        if grad.len() != self.size {
            return Err(MlErr::SizeMismatch {
                what: "grad",
                got: grad.len(),
                expected: self.size,
            });
        }

        let (f, ..) = self.kernel_dim;
        let w_size = self.w_size;

        // SAFETY: The if condition above checks that the size of the
        //         gradient is exactly the size of the layer.
        let (dw_raw, db_raw) = grad.split_at_mut(w_size);
        let dw = ArrayViewMut4::from_shape(self.kernel_dim, dw_raw).unwrap();
        let db = ArrayViewMut1::from_shape(f, db_raw).unwrap();

        Ok((dw, db))
    }
}

#[cfg(test)]
mod tests {
    use std::env;

    use super::*;

    #[test]
    fn test_conv2d_forward_backward_consistency() {
        unsafe { env::set_var("RUST_BACKTRACE", "1") };

        let mut layer = Conv2d::new(
            1,      // 1 filter
            1,      // 1 input channel
            (2, 2), // 2x2 kernel
            2,      // stride
            0,      // no padding
        );

        let input = Array4::from_elem((1, 1, 4, 4), 1.0);
        let params: Vec<_> = (0..layer.size()).map(|i| i as f32 / 10.0).collect();
        let mut grads = vec![0.0; layer.size()];

        let output = layer.forward(&params, input.view()).unwrap();
        assert_eq!(output.dim(), (1, 1, 2, 2));

        println!("{output:?}");

        let d_out = Array4::from_elem((1, 1, 2, 2), 1.0);
        layer
            .backward(&params, &mut grads, d_out.view().to_owned().view_mut())
            .unwrap();

        assert!((grads[4] - 4.0).abs() < 1e-5);
        println!("Gradient: {:?}", grads);
    }

    #[test]
    fn test_conv2d_forward() {
        unsafe { env::set_var("RUST_BACKTRACE", "1") };

        let input: Array4<f32> = array![[[
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0]
        ]]];

        let filters = 1;
        let in_channels = 1;
        let kernel_size = (2, 2);
        let stride = 2;
        let padding = 0;
        let mut conv = Conv2d::new(filters, in_channels, kernel_size, stride, padding);

        let params: [f32; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
        let expected_output = array![[[[31.0, 51.0], [111.0, 131.0]]]];

        let output = conv.forward(&params[..], input.view()).unwrap();

        assert_eq!(output, expected_output);
    }

    #[test]
    fn test_dilate_and_pad() {
        let filters = 1;
        let in_channels = 1;
        let kernel_size = (2, 2);
        let stride = 2;
        let padding = 0;
        let mut conv = Conv2d::new(filters, in_channels, kernel_size, stride, padding);

        let delta: Array4<f32> = array![[[
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0]
        ]]];

        let expected = array![[[
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
            [0.0, 5.0, 0.0, 6.0, 0.0, 7.0, 0.0, 8.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
            [0.0, 9.0, 0.0, 10.0, 0.0, 11.0, 0.0, 12.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
            [0.0, 13.0, 0.0, 14.0, 0.0, 15.0, 0.0, 16.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ]]];

        conv.dilate_and_pad(delta.view()).unwrap();
        let got = conv.dilated;

        assert_eq!(got, expected);
    }

    #[test]
    fn test_conv2d_backward() {
        unsafe { env::set_var("RUST_BACKTRACE", "1") };

        let input: Array4<f32> = array![[[
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0]
        ]]];

        let filters = 1;
        let in_channels = 1;
        let kernel_size = (2, 2);
        let stride = 2;
        let padding = 0;
        let mut conv = Conv2d::new(filters, in_channels, kernel_size, stride, padding);

        let params: [f32; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut grad: [f32; 5] = [0.0, 0.0, 0.0, 0.0, 0.0];
        let _ = conv.forward(&params, input.view()).unwrap();

        let mut delta_in: Array4<f32> = array![[[[17.0, 18.0], [19.0, 20.0]]]];

        let expected_delta_out = array![[[
            [68.0, 51.0, 72.0, 54.0],
            [34.0, 17.0, 36.0, 18.0],
            [76.0, 57.0, 80.0, 60.0],
            [38.0, 19.0, 40.0, 20.0]
        ]]];

        let expected_grad = [426.0, 500.0, 722.0, 796.0, 74.0];

        let delta_out = conv
            .backward(&params, &mut grad, delta_in.view_mut())
            .unwrap();

        assert_eq!(delta_out, expected_delta_out);
        assert_eq!(grad, expected_grad);
    }
}
