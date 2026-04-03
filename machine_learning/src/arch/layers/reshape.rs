use ndarray::{
    Array, ArrayBase, ArrayD, ArrayView2, ArrayView4, ArrayViewD, ArrayViewMut2, ArrayViewMut4,
    ArrayViewMutD, Data, Dimension, Ix2, Ix4, IxDyn,
};

use crate::{MlErr, Result, arch::InplaceReshape};

/// The metadata of a `Reshape` layer.
#[derive(Clone)]
pub struct Reshape2 {
    buf: ArrayD<f32>,
    channels: usize,
    height: usize,
    width: usize,
}

/// A `Reshape` layer.
#[derive(Clone)]
pub enum Reshape {
    TwoDTo4D(Reshape2),
    FourDTo2D(Reshape2),
}

impl Reshape {
    pub fn two_d_to4d(channels: usize, height: usize, width: usize) -> Self {
        let zeros = ArrayD::zeros(IxDyn(&[0]));

        Self::TwoDTo4D(Reshape2 {
            buf: zeros,
            channels,
            height,
            width,
        })
    }

    pub fn four_d_to2d(channels: usize, height: usize, width: usize) -> Self {
        let zeros = ArrayD::zeros(IxDyn(&[0]));

        Self::FourDTo2D(Reshape2 {
            buf: zeros,
            channels,
            height,
            width,
        })
    }

    pub fn size(&self) -> usize {
        0
    }

    /// Converts a tensor's dimensionality to the layer's output dimensionality.
    ///
    /// # Args
    /// * `input` - The tensor whose dimensionality is to be converted.
    ///
    /// # Returns
    /// The tensor with its dimensionality converted to match the layer's output.
    pub fn forward<'a>(&'a mut self, input: ArrayViewD<f32>) -> Result<ArrayViewD<'a, f32>> {
        match self {
            Self::TwoDTo4D(reshape) => {
                let input = input.into_dimensionality::<Ix2>().unwrap();
                reshape.two_d_to_4d(input)?;
                let reshaped = reshape.two_d_to_4d(input.view())?;

                Ok(reshaped.into_dyn())
            }
            Self::FourDTo2D(reshape) => {
                let input = input.into_dimensionality().unwrap();
                let reshaped = reshape.four_d_to_2d(input)?;

                Ok(reshaped.into_dyn())
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
    pub fn backward<'a>(&'a mut self, delta: ArrayViewMutD<f32>) -> Result<ArrayViewMutD<'a, f32>> {
        match self {
            Self::TwoDTo4D(reshape) => {
                let delta = delta.into_dimensionality().unwrap();
                let reshaped = reshape.four_d_to_2d_mut(delta.view())?;

                Ok(reshaped.into_dyn())
            }
            Self::FourDTo2D(reshape) => {
                let delta = delta.into_dimensionality().unwrap();
                let reshaped = reshape.two_d_to_4d_mut(delta.view())?;

                Ok(reshaped.into_dyn())
            }
        }
    }
}

/* Finally went with the reshape buffer
 * TODO: avoid code repetition, the problem here is that there doesn't exist a "downgrade" method
 * for converting an array view mut into just a view, but there's probably still code that is
 * unecessary dupped
 * TODO: maybe check the other solution: having the dataset split its sample and label memory
 * ***/
impl Reshape2 {
    fn two_d_to_4d<'a>(&'a mut self, arr: ArrayView2<f32>) -> Result<ArrayView4<'a, f32>>
where {
        let Reshape2 {
            ref mut buf,
            channels,
            height,
            width,
        } = *self;
        let batch_size = arr.dim().0;

        buf.reshape_inplace(arr.raw_dim().into_dyn());
        buf.assign(&arr);

        let arr_size = arr.dim().1;
        buf.view()
            .into_shape_with_order((batch_size, channels, height, width))
            .map_err(|_| MlErr::SizeMismatch {
                what: "2d array to 4d array",
                got: arr_size,
                expected: height * width,
            })
    }

    fn four_d_to_2d<'a>(&'a mut self, arr: ArrayView4<f32>) -> Result<ArrayView2<'a, f32>>
where {
        let Reshape2 {
            ref mut buf,
            channels,
            height,
            width,
        } = *self;
        let batch_size = arr.dim().0;

        buf.reshape_inplace(arr.raw_dim().into_dyn());
        buf.assign(&arr);

        let arr_size = arr.dim().2 * arr.dim().3;
        buf.view()
            .into_shape_with_order((batch_size, channels * height * width))
            .map_err(|_| MlErr::SizeMismatch {
                what: "4d array to 2d array",
                got: arr_size,
                expected: height * width,
            })
    }

    fn two_d_to_4d_mut<'a>(&'a mut self, arr: ArrayView2<f32>) -> Result<ArrayViewMut4<'a, f32>>
where {
        let Reshape2 {
            ref mut buf,
            channels,
            height,
            width,
        } = *self;
        let batch_size = arr.dim().0;

        buf.reshape_inplace(arr.raw_dim().into_dyn());
        buf.assign(&arr);

        let arr_size = arr.dim().1;
        buf.view_mut()
            .into_shape_with_order((batch_size, channels, height, width))
            .map_err(|_| MlErr::SizeMismatch {
                what: "2d array to 4d array",
                got: arr_size,
                expected: height * width,
            })
    }

    fn four_d_to_2d_mut<'a>(&'a mut self, arr: ArrayView4<f32>) -> Result<ArrayViewMut2<'a, f32>>
where {
        let Reshape2 {
            ref mut buf,
            channels,
            height,
            width,
        } = *self;
        let batch_size = arr.dim().0;

        buf.reshape_inplace(arr.raw_dim().into_dyn());
        buf.assign(&arr);

        let arr_size = arr.dim().2 * arr.dim().3;
        buf.view_mut()
            .into_shape_with_order((batch_size, channels * height * width))
            .map_err(|_| MlErr::SizeMismatch {
                what: "4d array to 2d array",
                got: arr_size,
                expected: height * width,
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_reshape_two_d_to4d_forward() {
        let channels = 1;
        let height = 2;
        let width = 2;
        let mut reshape = Reshape::two_d_to4d(channels, height, width);

        let input = array![[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.]];
        let expected = array![
            [[[1.0, 2.0], [3.0, 4.0]]],
            [[[5.0, 6.0], [7.0, 8.0]]],
            [[[9.0, 10.0], [11.0, 12.0]]]
        ]
        .into_dyn();
        let output = reshape.forward(input.into_dyn().view()).unwrap();

        assert_eq!(output, expected);
    }

    #[test]
    fn test_reshape_two_d_to4d_backward() {
        let channels = 1;
        let height = 2;
        let width = 2;
        let mut reshape = Reshape::two_d_to4d(channels, height, width);

        let input = array![
            [[[1.0, 2.0], [3.0, 4.0]]],
            [[[5.0, 6.0], [7.0, 8.0]]],
            [[[9.0, 10.0], [11.0, 12.0]]]
        ]
        .into_dyn();
        let expected = array![[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.]].into_dyn();
        let output = reshape.backward(input.into_dyn().view_mut()).unwrap();

        assert_eq!(output, expected);
    }

    #[test]
    fn test_reshape_four_d_to_2d_forward() {
        let channels = 1;
        let height = 2;
        let width = 2;
        let mut reshape = Reshape::four_d_to2d(channels, height, width);

        let input = array![
            [[[1.0, 2.0], [3.0, 4.0]]],
            [[[5.0, 6.0], [7.0, 8.0]]],
            [[[9.0, 10.0], [11.0, 12.0]]]
        ]
        .into_dyn();
        let expected = array![[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.]].into_dyn();
        let output = reshape.forward(input.into_dyn().view()).unwrap();

        assert_eq!(output, expected);
    }

    #[test]
    fn test_reshape_four_d_to2d_backward() {
        let channels = 1;
        let height = 2;
        let width = 2;
        let mut reshape = Reshape::four_d_to2d(channels, height, width);

        let input = array![[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.]];
        let expected = array![
            [[[1.0, 2.0], [3.0, 4.0]]],
            [[[5.0, 6.0], [7.0, 8.0]]],
            [[[9.0, 10.0], [11.0, 12.0]]]
        ]
        .into_dyn();
        let output = reshape.backward(input.into_dyn().view_mut()).unwrap();

        assert_eq!(output, expected);
    }
}
