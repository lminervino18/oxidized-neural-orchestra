use std::cmp;

use ndarray::prelude::*;

use crate::{Result, arch::InplaceReshape};

#[derive(Clone, Debug)]
pub struct MaxPooling {
    /// The size of the square filter matrix.
    filter_size: usize,
    stride: usize,
    padding: usize,

    // Forward metadata
    real_input_dim: (usize, usize),
    /// The input that's actually used during the forward convolution
    effective_input: Array4<f32>,
    output: Array4<f32>,

    // Backward metadata
    delta_out: Array4<f32>,
    max_indices: Array4<(usize, usize)>,
}

impl MaxPooling {
    pub fn new(filter_size: usize, stride: usize, padding: usize) -> Self {
        let real_input_dim = (0, 0);

        let zeros4 = Array4::zeros((1, 1, 1, 1));
        let max_indices = Array4::from_elem((1, 1, 1, 1), (0, 0));

        Self {
            filter_size,
            stride,
            padding,
            real_input_dim,
            effective_input: zeros4.clone(),
            output: zeros4.clone(),
            delta_out: zeros4.clone(),
            max_indices,
        }
    }

    pub fn size(&self) -> usize {
        0
    }

    pub fn forward(&mut self, x: ArrayView4<f32>) -> Result<ArrayView4<'_, f32>> {
        let Self {
            filter_size,
            stride,
            padding,
            ref mut real_input_dim,
            ref mut effective_input,
            ref mut output,
            ref mut max_indices,
            ..
        } = *self;

        let (batch_size, filters, input_height, input_width) = x.dim();
        *real_input_dim = (input_height, input_width);

        let output_height = (input_height + 2 * padding - filter_size) / stride + 1;
        let output_width = (input_width + 2 * padding - filter_size) / stride + 1;

        let effective_height = (output_height - 1) * stride + filter_size;
        let effective_width = (output_width - 1) * stride + filter_size;

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
        max_indices.reshape_inplace((batch_size, filters, output_height, output_width));

        for b in 0..batch_size {
            let input_b = effective_input.index_axis(Axis(0), b);
            let mut output_b = output.index_axis_mut(Axis(0), b);
            let mut max_indices_b = max_indices.index_axis_mut(Axis(0), b);

            for f in 0..filters {
                let input_bf = input_b.index_axis(Axis(0), f);
                let mut output_bf = output_b.index_axis_mut(Axis(0), f);
                let mut max_indices_bf = max_indices_b.index_axis_mut(Axis(0), f);

                for h in 0..output_height {
                    for w in 0..output_width {
                        let chunk_origin = (h * stride, w * stride);

                        let input_chunk = input_bf.slice(s![
                            chunk_origin.0..chunk_origin.0 + filter_size,
                            chunk_origin.1..chunk_origin.1 + filter_size,
                        ]);

                        // SAFETY: `filter_size` is a non-zero positive integer, so there must be
                        //         at least one element.
                        let (mut max_idx, max) = input_chunk
                            .indexed_iter()
                            .max_by(|(_, a), (_, b)| a.total_cmp(b))
                            .unwrap();

                        max_idx = (max_idx.0 + chunk_origin.0, max_idx.1 + chunk_origin.1);

                        output_bf[[h, w]] = *max;
                        max_indices_bf[[h, w]] = max_idx;
                    }
                }
            }
        }

        Ok(output.view())
    }

    pub fn backward(&mut self, d_in: ArrayViewMut4<f32>) -> Result<ArrayViewMut4<'_, f32>> {
        let Self {
            real_input_dim,
            ref mut delta_out,
            ref max_indices,
            ..
        } = *self;

        let (batches, in_channels, _, _) = d_in.dim();
        delta_out.reshape_inplace((batches, in_channels, real_input_dim.0, real_input_dim.1));
        delta_out.fill(0.);

        for b in 0..batches {
            for c in 0..in_channels {
                azip!((d in &d_in, &idx in max_indices) {
                    delta_out[[b,c,idx.0,idx.1]] = *d;
                });
            }
        }

        Ok(delta_out.view_mut())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_pooling00_forward_with_no_padding_stride1_and_filter_size1_equals_input() {
        let input = array![[[
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.],
            [13., 14., 15., 16.]
        ]]];
        let mut max_pooling = MaxPooling::new(1, 1, 0);

        let expected = &input;

        let output = max_pooling.forward(input.view()).unwrap();

        assert_eq!(output, expected);
    }

    #[test]
    fn test_max_pooling01_forward_1batch_1filter_4_by_4_img_filter_size_2_stride2_padding0() {
        let input = array![[[
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.],
            [13., 14., 15., 16.]
        ]]];
        let mut max_pooling = MaxPooling::new(2, 2, 0);

        let expected = array![[[[6., 8.], [14., 16.]]]];

        let output = max_pooling.forward(input.view()).unwrap();

        assert_eq!(output, expected);
    }

    #[test]
    fn test_max_pooling02_backward_with_no_padding_stride1_and_filter_size1_equals_input() {
        let input = array![[[
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.],
            [13., 14., 15., 16.]
        ]]];
        let mut max_pooling = MaxPooling::new(1, 1, 0);
        max_pooling.forward(input.view()).unwrap();
        let mut delta_in = array![[[
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.],
            [13., 14., 15., 16.]
        ]]];

        let output = max_pooling.backward(delta_in.view_mut()).unwrap();
        let expected = &delta_in;

        assert_eq!(output, expected);
    }

    #[test]
    fn test_max_pooling03_backward_1batch_1filter_4_by_4_img_filter_size_2_stride2_padding0() {
        let input = array![[[
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.],
            [13., 14., 15., 16.]
        ]]];
        let mut max_pooling = MaxPooling::new(2, 2, 0);
        max_pooling.forward(input.view()).unwrap();
        let mut delta_in = array![[[[6., 8.], [14., 16.]]]];

        let expected_max_indices = array![[[[(1, 1), (1, 3)], [(3, 1), (3, 3)]]]];
        assert_eq!(max_pooling.max_indices, expected_max_indices);

        let delta_in = max_pooling.backward(delta_in.view_mut()).unwrap();
        let expected_delta_in = array![[[
            [0., 0., 0., 0.],
            [0., 6., 0., 8.],
            [0., 0., 0., 0.],
            [0., 14., 0., 16.]
        ]]];

        assert_eq!(delta_in, expected_delta_in);
    }

    #[test]
    fn test_max_pooling04_stride1_filter_size2() {
        unsafe {
            std::env::set_var("RUST_BACKTRACE", "1");
        }

        let input = array![[[
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.],
            [13., 14., 15., 16.]
        ]]];
        let mut max_pooling = MaxPooling::new(2, 1, 0);
        max_pooling.forward(input.view()).unwrap();
        let mut delta_in = array![[[[6., 7., 8.], [10., 11., 12.], [14., 15., 16.]]]];

        let expected_max_indices = array![[[
            [(1, 1), (1, 2), (1, 3)],
            [(2, 1), (2, 2), (2, 3)],
            [(3, 1), (3, 2), (3, 3)]
        ]]];
        assert_eq!(max_pooling.max_indices, expected_max_indices);

        let output = max_pooling.backward(delta_in.view_mut()).unwrap();
        let expected = array![[[
            [0., 0., 0., 0.],
            [0., 6., 7., 8.],
            [0., 10., 11., 12.],
            [0., 14., 15., 16.]
        ]]];

        assert_eq!(output, expected);
    }

    #[test]
    fn test_max_pooling04_rectangle_input() {
        unsafe {
            std::env::set_var("RUST_BACKTRACE", "1");
        }

        let input = array![[[
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.],
            [13., 14., 15., 16.],
            [17., 18., 19., 20.]
        ]]];
        let mut max_pooling = MaxPooling::new(2, 1, 0);
        max_pooling.forward(input.view()).unwrap();
        let mut delta_in = array![[[
            [6., 7., 8.],
            [10., 11., 12.],
            [14., 15., 16.],
            [18., 19., 20.]
        ]]];

        let expected_max_indices = array![[[
            [(1, 1), (1, 2), (1, 3)],
            [(2, 1), (2, 2), (2, 3)],
            [(3, 1), (3, 2), (3, 3)],
            [(4, 1), (4, 2), (4, 3)]
        ]]];
        assert_eq!(max_pooling.max_indices, expected_max_indices);

        let output = max_pooling.backward(delta_in.view_mut()).unwrap();
        let expected = array![[[
            [0., 0., 0., 0.],
            [0., 6., 7., 8.],
            [0., 10., 11., 12.],
            [0., 14., 15., 16.],
            [0., 18., 19., 20.]
        ]]];

        assert_eq!(output, expected);
    }
}
