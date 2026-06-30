use ndarray::{Array2, Array3, ArrayView2, ArrayView3, ArrayView4, ArrayViewMut3, linalg, s};

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
    pub fn conv_im2col_into(
        &self,
        buf: &mut ArrayViewMut3<f32>,
        image: ArrayView3<f32>,
        kernel: ArrayView4<f32>,
        _reverse: bool, // wip
        stride: usize,
    ) {
        let (channels, image_h, image_w) = image.dim();
        let (filters, in_channels, kernel_w, kernel_h) = kernel.dim();
        assert_eq!(channels, in_channels);
        assert_eq!(kernel_w, kernel_h);

        let kernel_size = kernel_w;
        let col_image = self.im2col(image, kernel_size, stride);

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

    pub fn conv_col2im_into(
        &self,
        buf: &mut ArrayViewMut3<f32>,
        delta: ArrayView3<f32>,
        kernel: ArrayView4<f32>,
        _reverse: bool, // wip
        stride: usize,
    ) {
        let (filters2, delta_h, delta_w) = delta.dim();
        let (filters, in_channels, kernel_w, kernel_h) = kernel.dim();
        assert_eq!(filters, filters2); // TODO: esto lo saco
        assert_eq!(kernel_w, kernel_h);

        let kernel_size = kernel_h;

        // SAFETY: `kernel` has filters * in_channels * kernel_size^2 elements.
        let col_kernel = kernel
            .into_shape_with_order((filters, in_channels * kernel_size * kernel_size))
            .unwrap();
        // SAFETY: `delta` has filters * delta_h * delta_w elements.
        let col_delta = delta
            .into_shape_with_order((filters, delta_h * delta_w))
            .unwrap();

        // TODO: poner esto en metadata o algo
        let mut col_delta_out =
            Array2::zeros((in_channels * kernel_size * kernel_size, delta_h * delta_w));

        linalg::general_mat_mul(1.0, &col_kernel.t(), &col_delta, 0.0, &mut col_delta_out);

        // buf dim es porque estoy suponiendo q ya reshapearon afuera, lo podría escribir mejor
        // quizás
        let delta_out = self.col2im(col_delta_out.view(), buf.dim(), kernel_size, stride);
        buf.assign(&delta_out);
    }

    // TODO: no es técnicamente un reshape porq estoy básicamente copiando píxeles q no habría si no
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
    pub fn im2col(&self, image: ArrayView3<f32>, kernel_size: usize, stride: usize) -> Array2<f32> {
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

    pub fn col2im(
        &self,
        col_delta: ArrayView2<f32>,
        image_shape: (usize, usize, usize),
        kernel_size: usize,
        stride: usize,
    ) -> Array3<f32> {
        let (_, image_h, image_w) = image_shape;

        let out_w = (image_w - kernel_size) / stride + 1;

        // TODO: poner esto en metadata o algo
        let mut image = Array3::<f32>::zeros(image_shape);

        for (row_idx, i) in (0..image_h - kernel_size + 1).step_by(stride).enumerate() {
            for (col_idx, j) in (0..image_w - kernel_size + 1).step_by(stride).enumerate() {
                let col = col_delta.column(row_idx * out_w + col_idx);
                let mut window = image.slice_mut(s![.., i..i + kernel_size, j..j + kernel_size]);

                window.iter_mut().zip(col.iter()).for_each(|(w, c)| {
                    *w += *c;
                })
            }
        }

        image
    }
}
