use ndarray::{ArrayView2, ArrayView4, ArrayViewD, Ix2};

/// The metadata of a `Reshape` layer.
pub struct Reshape2 {
    channels: usize,
    height: usize,
    width: usize,
}

/// A `Reshape` layer.
pub enum Reshape {
    TwoDTo4D(Reshape2),
    FourDTo2D(Reshape2),
}

impl Reshape {
    /// Converts a tensor's dimensionality to the layer's output dimensionality.
    ///
    /// # Args
    /// * `input` - The tensor whose dimensionality is to be converted.
    ///
    /// # Returns
    /// The tensor with its dimensionality converted to match the layer's output.
    fn forward<'a>(&self, input: ArrayViewD<'a, u32>) -> ArrayViewD<'a, u32> {
        match self {
            Self::TwoDTo4D(reshape) => {
                let input = input.into_dimensionality::<Ix2>().unwrap();

                reshape.two_d_to_4d(input).into_dyn()
            }
            Self::FourDTo2D(reshape) => {
                let input = input.into_dimensionality().unwrap();

                reshape.four_d_to_2d(input).into_dyn()
            }
        }
    }

    /// Converts a tensor's dimensionality to the layer's input dimensionality.
    ///
    /// # Args
    /// * `delta` - The tensor whose dimensionality is to be converted.
    ///
    /// # Returns
    /// The tensor with its dimensionality converted to match the layer's input.
    fn backward<'a>(&self, delta: ArrayViewD<'a, u32>) -> ArrayViewD<'a, u32> {
        match self {
            Self::TwoDTo4D(reshape) => {
                let delta = delta.into_dimensionality().unwrap();

                reshape.four_d_to_2d(delta).into_dyn()
            }
            Self::FourDTo2D(reshape) => {
                let delta = delta.into_dimensionality::<Ix2>().unwrap();

                reshape.two_d_to_4d(delta).into_dyn()
            }
        }
    }
}

impl Reshape2 {
    fn two_d_to_4d<'a>(&self, arr: ArrayView2<'a, u32>) -> ArrayView4<'a, u32> {
        let Reshape2 {
            channels,
            height,
            width,
        } = *self;
        let batch_size = arr.dim().0;

        arr.into_shape_with_order((batch_size, channels, height, width))
            .unwrap()
    }

    fn four_d_to_2d<'a>(&self, arr: ArrayView4<'a, u32>) -> ArrayView2<'a, u32> {
        let Reshape2 {
            channels,
            height,
            width,
        } = *self;
        let batch_size = arr.dim().0;

        arr.into_shape_with_order((batch_size, channels * height * width))
            .unwrap()
    }
}
