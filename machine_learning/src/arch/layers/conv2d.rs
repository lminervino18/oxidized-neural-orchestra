use ndarray::{
    Array2, Array4, ArrayView1, ArrayView4, ArrayViewMut1, ArrayViewMut4, Axis, linalg, s,
};

use crate::{MlErr, Result, arch::InplaceReshape};

#[derive(Clone)]
pub struct Conv2d {
    kernel_dim: (usize, usize, usize, usize),
    stride: usize,
    padding: usize,
    size: usize,
    w_size: usize,

    // Forward metadata
    input_padded: Array4<f32>,
    columns: Array2<f32>,
    output_buffer: Array4<f32>,
    res_buf: Array2<f32>,

    // Output metadata
    input: Array4<f32>,
    delta: Array4<f32>,
    d_cols: Array2<f32>,
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
        let zeros2 = Array2::zeros((1, 1));
        let zeros4 = Array4::zeros((1, 1, 1, 1));

        Self {
            kernel_dim: (filters, in_channels, kh, kw),
            stride,
            padding,
            size,
            w_size,
            input_padded: zeros4.clone(),
            columns: zeros2.clone(),
            res_buf: zeros2.clone(),
            output_buffer: zeros4.clone(),
            input: zeros4.clone(),
            delta: zeros4.clone(),
            d_cols: zeros2.clone(),
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn forward(&mut self, params: &[f32], x: ArrayView4<f32>) -> Result<ArrayView4<'_, f32>> {
        let (f, c, kh, kw) = self.kernel_dim;
        let (batch, _, h, w) = x.dim();

        let out_h = (h + 2 * self.padding - kh) / self.stride + 1;
        let out_w = (w + 2 * self.padding - kw) / self.stride + 1;

        self.input.reshape_inplace(x.raw_dim());
        self.input.assign(&x);

        self.im2col(x);

        let (w, b) = self.view_params(params)?;
        let w_flat = w.to_shape((f, c * kh * kw)).unwrap();

        self.res_buf.reshape_inplace((f, batch * out_h * out_w));
        linalg::general_mat_mul(1.0, &w_flat, &self.columns, 0.0, &mut self.res_buf);
        self.res_buf += &b.insert_axis(Axis(1));

        let temp_view = self.res_buf.to_shape((f, out_h, out_w, batch)).unwrap();
        let permuted_axes = temp_view.permuted_axes([3, 0, 1, 2]);
        self.output_buffer.reshape_inplace((batch, f, out_h, out_w));
        self.output_buffer.assign(&permuted_axes);

        Ok(self.output_buffer.view())
    }

    pub fn backward(
        &mut self,
        params: &[f32],
        grad: &mut [f32],
        d: ArrayViewMut4<f32>,
    ) -> Result<ArrayViewMut4<'_, f32>> {
        let (f, c, kh, kw) = self.kernel_dim;
        let (batch, _, xh, xw) = self.input.dim();
        let out_h = (xh + 2 * self.padding - kh) / self.stride + 1;
        let out_w = (xw + 2 * self.padding - kw) / self.stride + 1;

        let (mut dw, mut db) = self.view_grad(grad)?;
        let (w, _) = self.view_params(params)?;

        let permuted = d.permuted_axes([1, 2, 3, 0]);
        let d_out_reshaped = permuted.to_shape((f, batch * out_h * out_w)).unwrap();

        let db_sum = d_out_reshaped.sum_axis(Axis(1));
        db.assign(&db_sum);

        let dw_view = dw.view_mut();
        let mut dw_flat = dw_view.to_shape((f, c * kh * kw)).unwrap();
        linalg::general_mat_mul(1.0, &d_out_reshaped, &self.columns.t(), 0.0, &mut dw_flat);

        let w_view = w.view();
        let w_flat = w_view.to_shape((f, c * kh * kw)).unwrap();

        self.d_cols
            .reshape_inplace((c * kh * kw, batch * out_h * out_w));
        linalg::general_mat_mul(1.0, &w_flat.t(), &d_out_reshaped, 0.0, &mut self.d_cols);

        self.col2im();
        Ok(self.delta.view_mut())
    }

    fn im2col(&mut self, x: ArrayView4<f32>) {
        let (batch, channels, h, w) = x.dim();
        let (_, _, kh, kw) = self.kernel_dim;

        let double_padding = self.padding << 1;
        let p_h = h + double_padding;
        let p_w = w + double_padding;
        let out_h = (p_h - kh) / self.stride + 1;
        let out_w = (p_w - kw) / self.stride + 1;

        self.input_padded
            .reshape_inplace((batch, channels, p_h, p_w));
        self.input_padded.fill(0.0);
        self.input_padded
            .slice_mut(s![
                ..,
                ..,
                self.padding..h + self.padding,
                self.padding..w + self.padding
            ])
            .assign(&x);

        let patch_size = channels * kh * kw;
        let num_patches = batch * out_h * out_w;
        self.columns.reshape_inplace((patch_size, num_patches));

        let mut col_idx = 0;
        for y in (0..out_h).map(|i| i * self.stride) {
            for x_coord in (0..out_w).map(|i| i * self.stride) {
                for b in 0..batch {
                    let window =
                        self.input_padded
                            .slice(s![b, .., y..y + kh, x_coord..x_coord + kw]);

                    self.columns
                        .column_mut(col_idx)
                        .assign(&window.to_shape(patch_size).unwrap());

                    col_idx += 1;
                }
            }
        }
    }

    fn col2im(&mut self) {
        let (batch, channels, h, w) = self.input.dim();
        let (_, _, kh, kw) = self.kernel_dim;
        let out_h = (h + 2 * self.padding - kh) / self.stride + 1;
        let out_w = (w + 2 * self.padding - kw) / self.stride + 1;

        let p_h = h + 2 * self.padding;
        let p_w = w + 2 * self.padding;
        self.input_padded
            .reshape_inplace((batch, channels, p_h, p_w));
        self.input_padded.fill(0.0);

        let mut col_idx = 0;
        for y in (0..out_h).map(|i| i * self.stride) {
            for x_coord in (0..out_w).map(|i| i * self.stride) {
                for b in 0..batch {
                    let patch = self.d_cols.column(col_idx);
                    let patch_reshaped = patch.to_shape((channels, kh, kw)).unwrap();

                    let mut target_slice =
                        self.input_padded
                            .slice_mut(s![b, .., y..y + kh, x_coord..x_coord + kw]);
                    target_slice += &patch_reshaped;

                    col_idx += 1;
                }
            }
        }

        self.delta.reshape_inplace((batch, channels, h, w));
        if self.padding > 0 {
            self.delta.assign(&self.input_padded.slice(s![
                ..,
                ..,
                self.padding..h + self.padding,
                self.padding..w + self.padding
            ]));
        } else {
            self.delta.assign(&self.input_padded);
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
    use super::*;

    #[test]
    fn test_conv2d_forward_backward_consistency() {
        let mut layer = Conv2d::new(
            1,      // 1 filter
            1,      // 1 input channel
            (2, 2), // 2x2 kernel
            1,      // stride
            0,      // no padding
        );

        let input = Array4::from_elem((1, 1, 3, 3), 1.0);
        let params = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let mut grads = vec![0.0; 5];

        let output = layer.forward(&params, input.view()).unwrap();
        assert_eq!(output.dim(), (1, 1, 2, 2));

        let d_out = Array4::from_elem((1, 1, 2, 2), 1.0);
        layer
            .backward(&params, &mut grads, d_out.view().to_owned().view_mut())
            .unwrap();

        assert!((grads[4] - 4.0).abs() < 1e-5);
        println!("Gradients: {:?}", grads);
    }
}
