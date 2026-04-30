use ndarray::{ArrayView2, ArrayView4, ArrayViewD, ArrayViewMut2, ArrayViewMut4, ArrayViewMutD};

use crate::{MlErr, Result};

/// The metadata of a `Reshape` layer.
#[derive(Clone, Debug)]
pub struct TwoDTo4D {
    channels: usize,
    height: usize,
    width: usize,
}

#[derive(Clone, Debug)]
pub struct FourDTo2D {
    channels: usize,
    height: usize,
    width: usize,
}

/// A `Reshape` layer.
#[derive(Clone, Debug)]
pub enum Reshape {
    TwoDTo4D(TwoDTo4D),
    FourDTo2D(FourDTo2D),
}

impl Reshape {
    pub fn two_d_to4d(channels: usize, height: usize, width: usize) -> Self {
        Self::TwoDTo4D(TwoDTo4D {
            channels,
            height,
            width,
        })
    }

    pub fn four_d_to2d(channels: usize, height: usize, width: usize) -> Self {
        Self::FourDTo2D(FourDTo2D {
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
    pub fn forward<'a>(&self, input: ArrayViewD<'a, f32>) -> Result<ArrayViewD<'a, f32>> {
        match self {
            Self::TwoDTo4D(reshape) => {
                let arr = input.into_dimensionality()?;
                Ok(reshape.forward(arr)?.into_dyn())
            }
            Self::FourDTo2D(reshape) => {
                let arr = input.into_dimensionality()?;
                Ok(reshape.forward(arr)?.into_dyn())
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
    pub fn backward<'a>(&self, delta: ArrayViewMutD<'a, f32>) -> Result<ArrayViewMutD<'a, f32>> {
        match self {
            Self::TwoDTo4D(reshape) => {
                let arr = delta.into_dimensionality()?;
                Ok(reshape.backward(arr)?.into_dyn())
            }
            Self::FourDTo2D(reshape) => {
                let arr = delta.into_dimensionality()?;
                Ok(reshape.backward(arr)?.into_dyn())
            }
        }
    }
}

impl TwoDTo4D {
    fn forward<'a>(&self, arr: ArrayView2<'a, f32>) -> Result<ArrayView4<'a, f32>>
where {
        let Self {
            channels,
            height,
            width,
        } = *self;
        let batch_size = arr.dim().0;

        let arr_size = arr.dim().1;
        arr.into_shape_with_order((batch_size, channels, height, width))
            .map_err(|_| MlErr::SizeMismatch {
                what: "2d array to 4d array",
                got: arr_size,
                expected: height * width,
            })
    }

    fn backward<'a>(&self, arr: ArrayViewMut4<'a, f32>) -> Result<ArrayViewMut2<'a, f32>>
where {
        let Self {
            channels,
            height,
            width,
        } = *self;
        let batch_size = arr.dim().0;

        let arr_size = arr.dim().2 * arr.dim().3;
        arr.into_shape_with_order((batch_size, channels * height * width))
            .map_err(|_| MlErr::SizeMismatch {
                what: "4d array to 2d array in backward",
                got: arr_size,
                expected: height * width,
            })
    }
}

impl FourDTo2D {
    fn forward<'a>(&self, arr: ArrayView4<'a, f32>) -> Result<ArrayView2<'a, f32>>
where {
        let Self {
            channels,
            height,
            width,
        } = *self;
        let batch_size = arr.dim().0;

        let arr_size = arr.dim().2 * arr.dim().3;
        arr.into_shape_with_order((batch_size, channels * height * width))
            .map_err(|_| MlErr::SizeMismatch {
                what: "4d array to 2d array in forward",
                got: arr_size,
                expected: height * width,
            })
    }

    fn backward<'a>(&self, arr: ArrayViewMut2<'a, f32>) -> Result<ArrayViewMut4<'a, f32>>
where {
        let Self {
            channels,
            height,
            width,
        } = *self;
        let batch_size = arr.dim().0;

        let arr_size = arr.dim().1;
        arr.into_shape_with_order((batch_size, channels, height, width))
            .map_err(|_| MlErr::SizeMismatch {
                what: "2d array to 4d array in backward",
                got: arr_size,
                expected: height * width,
            })
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn test_reshape_two_d_to4d_forward() {
        let channels = 1;
        let height = 2;
        let width = 2;
        let reshape = Reshape::two_d_to4d(channels, height, width);

        let input = array![[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.]].into_dyn();
        let expected = array![
            [[[1.0, 2.0], [3.0, 4.0]]],
            [[[5.0, 6.0], [7.0, 8.0]]],
            [[[9.0, 10.0], [11.0, 12.0]]]
        ]
        .into_dyn();
        let output = reshape.forward(input.view()).unwrap();

        assert_eq!(output, expected);
    }

    #[test]
    fn test_reshape_two_d_to4d_backward() {
        let channels = 1;
        let height = 2;
        let width = 2;
        let reshape = Reshape::two_d_to4d(channels, height, width);

        let mut input = array![
            [[[1.0, 2.0], [3.0, 4.0]]],
            [[[5.0, 6.0], [7.0, 8.0]]],
            [[[9.0, 10.0], [11.0, 12.0]]]
        ]
        .into_dyn();
        let expected = array![[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.]].into_dyn();
        let output = reshape.backward(input.view_mut()).unwrap();

        assert_eq!(output, expected);
    }

    #[test]
    fn test_reshape_four_d_to_2d_forward() {
        let channels = 1;
        let height = 2;
        let width = 2;
        let reshape = Reshape::four_d_to2d(channels, height, width);

        let input = array![
            [[[1.0, 2.0], [3.0, 4.0]]],
            [[[5.0, 6.0], [7.0, 8.0]]],
            [[[9.0, 10.0], [11.0, 12.0]]]
        ]
        .into_dyn();
        let expected = array![[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.]].into_dyn();
        let output = reshape.forward(input.view()).unwrap();

        assert_eq!(output, expected);
    }

    #[test]
    fn test_reshape_four_d_to2d_backward() {
        let channels = 1;
        let height = 2;
        let width = 2;
        let reshape = Reshape::four_d_to2d(channels, height, width);

        let mut input = array![[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.]].into_dyn();
        let expected = array![
            [[[1.0, 2.0], [3.0, 4.0]]],
            [[[5.0, 6.0], [7.0, 8.0]]],
            [[[9.0, 10.0], [11.0, 12.0]]]
        ]
        .into_dyn();
        let output = reshape.backward(input.view_mut()).unwrap();

        assert_eq!(output, expected);
    }

    #[test]
    fn test_reshape_two_d_to4d_forward_consecutive() {
        let channels = 1;
        let height = 2;
        let width = 2;
        let reshape = Reshape::two_d_to4d(channels, height, width);

        let input1 = array![[1., 2., 3., 4.]].into_dyn();
        let expected1 = array![[[[1., 2.], [3., 4.]]]].into_dyn();
        let output1 = reshape.forward(input1.view()).unwrap();

        assert_eq!(output1, expected1);

        let input2 = array![[1., 2., 3., 4.], [5., 6., 7., 8.],].into_dyn();
        let expected2 = array![[[[1., 2.], [3., 4.]]], [[[5., 6.], [7., 8.]]],].into_dyn();
        let output2 = reshape.forward(input2.view()).unwrap();

        assert_eq!(output2, expected2);

        let input3 = array![[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.],].into_dyn();
        let expected3 = array![
            [[[1., 2.], [3., 4.]]],
            [[[5., 6.], [7., 8.]]],
            [[[9., 10.], [11., 12.]]],
        ]
        .into_dyn();
        let output3 = reshape.forward(input3.view()).unwrap();

        assert_eq!(output3, expected3);

        let input4 = array![
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.],
            [13., 14., 15., 16.]
        ]
        .into_dyn();
        let expected4 = array![
            [[[1., 2.], [3., 4.]]],
            [[[5., 6.], [7., 8.]]],
            [[[9., 10.], [11., 12.]]],
            [[[13., 14.], [15., 16.]]]
        ]
        .into_dyn();
        let output4 = reshape.forward(input4.view()).unwrap();

        assert_eq!(output4, expected4);
    }
}
