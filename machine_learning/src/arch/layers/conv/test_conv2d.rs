use super::*;
use approx::assert_abs_diff_eq;
use ndarray::{Array4, array};
use std::env;

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
