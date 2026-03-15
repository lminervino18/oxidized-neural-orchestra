use ndarray::prelude::*;
use ndarray_conv::{ConvExt, ConvMode, PaddingMode};

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

        let (w, b) = self.view_params(params)?;
        let conv_mode = ConvMode::Custom {
            padding: [0, 0, padding, padding],
            strides: [1, 1, stride, stride],
        };

        let mut output = x.conv(&w, conv_mode, PaddingMode::Zeros).unwrap();
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
        let Self {
            padding,
            stride,
            kernel_dim: (.., kh, kw),
            ..
        } = *self;

        let (.., in_h, in_w) = self.input.dim();
        let (mut dw, mut db) = self.view_grad(grad)?;
        let (w, _) = self.view_params(params)?;

        let db_sum = d.sum_axis(Axis(0)).sum_axis(Axis(1)).sum_axis(Axis(1));
        db.assign(&db_sum);

        let (batch, filters, d_h, d_w) = d.dim();
        let dilated_shape = (
            batch,
            filters,
            (d_h - 1) * stride + 1,
            (d_w - 1) * stride + 1,
        );

        self.dilated.reshape_inplace(dilated_shape);
        self.dilated.fill(0.0);
        self.dilated
            .slice_mut(s![.., .., ..;stride, ..;stride])
            .assign(&d);

        let dw_mode = ConvMode::Custom {
            padding: [0, 0, padding, padding],
            strides: [1, 1, 1, 1],
        };

        let dw_conv = self
            .input
            .conv(&self.dilated, dw_mode, PaddingMode::Zeros)
            .unwrap();

        dw.assign(&dw_conv.slice(s![.., .., ..kh, ..kw]));

        let flipped = w.slice(s![.., .., ..;-1, ..;-1]);
        let full_delta = self
            .dilated
            .conv(&flipped, ConvMode::Full, PaddingMode::Zeros)
            .unwrap();

        let h_start = padding;
        let w_start = padding;
        let cropped_delta =
            full_delta.slice(s![.., .., h_start..h_start + in_h, w_start..w_start + in_w]);

        self.delta.reshape_inplace(cropped_delta.raw_dim());
        self.delta.assign(&cropped_delta);

        Ok(self.delta.view_mut())
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
}
