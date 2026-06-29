use crate::Result;

use ndarray::{Array2, Array3, ArrayView2, ArrayView3, ArrayView4, ArrayViewMut3, linalg, s};
use ndarray_conv::{ConvExt, ConvMode, PaddingMode, ReverseKernel};

#[derive(Clone, Debug)]
pub struct Convolver {
    // processor: Option<P>,
    // kernel_size: usize,
}

impl Convolver {
    pub fn new(_kernel_size: usize, _processor: Option<()>) -> Self {
        Self {}
    }

    /// Computes the convolution of the passed in kernel over input and writes it into the buffer.
    ///
    /// # Args
    /// * `buf` - The buffer for the result.
    /// * `input` - The **already padded** input tensor that is to be convolved.
    /// * `kernel` - The kernel that is to be used in the convolution.
    /// * `reverse` - Whether to reverse the kernel for the convolution, if not, then it becomes a
    ///   correlate operation.
    /// * `stride` - The size of the kernel steps.
    pub fn conv_into(
        &self,
        buf: &mut ArrayViewMut3<f32>,
        input: ArrayView3<f32>,
        kernel: ArrayView4<f32>,
        _reverse: bool, // wip
        stride: usize,
    ) {
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

        // buf.reshape_inplace((filters, out_h, out_w));
        // TODO: este safety va a ser "me lo reshapean afuera"
        // SAFETY: `buf` was just reshaped to have enough elements.
        let mut buf = buf
            .view_mut()
            .into_shape_with_order((filters, out_h * out_w))
            .unwrap();

        linalg::general_mat_mul(1.0, &col_kernel, &col_image, 0.0, &mut buf);
    }

    /// Reshapes a three-dimension image tensor into a matrix with columns that match the window
    /// views that the kernel sees during the convolution pass.
    ///
    /// # Args
    /// * `image` - The **already padded** image tensor.
    /// * `kernel_size` - The size of the square kernel.
    /// * `stride` - The size of the kernel steps.
    ///
    /// # Returns
    /// The reshaped matrix from the image tensor.
    fn im2col(&self, image: ArrayView3<f32>, kernel_size: usize, stride: usize) -> Array2<f32> {
        let (channels, image_h, image_w) = image.dim();

        let out_h = (image_h - kernel_size) / stride + 1;
        let out_w = (image_w - kernel_size) / stride + 1;

        // TODO: poner esto en metadata o algo
        let mut col_image = Array2::zeros((channels * kernel_size * kernel_size, out_h * out_w));

        for (row_idx, i) in (0..image_h - kernel_size + 1).step_by(stride).enumerate() {
            for (col_idx, j) in (0..image_w - kernel_size + 1).step_by(stride).enumerate() {
                let window = image.slice(s![.., i..i + kernel_size, j..j + kernel_size]);
                let mut col = col_image.column_mut(row_idx * out_w + col_idx);

                col.iter_mut().zip(window.iter()).for_each(|(c, w)| {
                    *c = *w;
                });
            }
        }

        col_image
    }

    // TODO: estos dos métodos vuelan
    pub fn conv2d(
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

    pub fn conv3d(
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
