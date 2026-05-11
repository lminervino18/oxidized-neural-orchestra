use std::cmp;

use ndarray::prelude::*;

use crate::{MlErr, Result, arch::InplaceReshape};

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
    dilated: Array4<f32>,
}

impl MaxPooling {
    pub fn new(filter_size: usize, stride: usize, padding: usize) -> Self {
        let real_input_dim = (0, 0);

        let zeros4 = Array4::zeros((1, 1, 1, 1));

        Self {
            filter_size,
            stride,
            padding,
            real_input_dim,
            effective_input: zeros4.clone(),
            output: zeros4.clone(),
            delta_out: zeros4.clone(),
            dilated: zeros4,
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

        for b in 0..batch_size {
            let input_b = self.effective_input.index_axis(Axis(0), b);

            for f in 0..filters {
                let input_bf = input_b.index_axis(Axis(0), f);

                for (x, i) in (0..effective_width).step_by(filter_size).enumerate() {
                    for (y, j) in (0..effective_width).step_by(filter_size).enumerate() {
                        let input_chunk =
                            input_bf.slice(s![i..i + filter_size, j..j + filter_size]);

                        // TODO: how could the iterator be empty?!!! add safety comment
                        let max = input_chunk.iter().max_by(|a, b| a.total_cmp(b)).unwrap();

                        output[[b, f, x, y]] = *max;
                    }
                }
            }
        }

        Ok(output.view())
    }

    pub fn backward(
        &mut self,
        params: &[f32],
        d_in: ArrayViewMut4<f32>,
    ) -> Result<ArrayViewMut4<'_, f32>> {
        todo!()
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
    fn test_max_pooling00_forward_1batch_1filter_4_by_4_img_filter_size_2_stride2_padding0() {
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
}
