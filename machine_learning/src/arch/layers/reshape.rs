use ndarray::{ArrayBase, ArrayViewD, ArrayViewMutD, Data, Ix2, Ix4};

use crate::{MlErr, Result};

/// The metadata of a `Reshape` layer.
#[derive(Clone)]
pub struct Reshape2 {
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
        Self::TwoDTo4D(Reshape2 {
            channels,
            height,
            width,
        })
    }

    pub fn four_d_to2d(channels: usize, height: usize, width: usize) -> Self {
        Self::FourDTo2D(Reshape2 {
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
                let input = input.into_dimensionality::<Ix2>().unwrap();

                Ok(reshape.two_d_to_4d(input)?.into_dyn())
            }
            Self::FourDTo2D(reshape) => {
                let input = input.into_dimensionality().unwrap();

                Ok(reshape.four_d_to_2d(input)?.into_dyn())
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
                let delta = delta.into_dimensionality().unwrap();

                Ok(reshape.four_d_to_2d(delta)?.into_dyn())
            }
            Self::FourDTo2D(reshape) => {
                let delta = delta.into_dimensionality::<Ix2>().unwrap();

                Ok(reshape.two_d_to_4d(delta)?.into_dyn())
            }
        }
    }
}

/* FIXME: this two methods are crashing because `into_shape_with_other()` is expecting memory to be
 * contiguous and this is not the case because of how we are handling the vecs for samples and
 * labels in `Dataset`. There are two options, the obvious and more tedious one would be to change
 * `Dataset` so that it has two different allocations for samples and labels, the second one would
 * be to have a buffer in which to copy the data in the `Reshape` layer. The latter sounds bad but
 * because it would be copying the whole dataset, but at least just once-allocated memory...
 ***/
impl Reshape2 {
    fn two_d_to_4d<S>(&self, arr: ArrayBase<S, Ix2>) -> Result<ArrayBase<S, Ix4>>
    where
        S: Data<Elem = f32>,
    {
        let Reshape2 {
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
                expected: self.height * self.width,
            })
    }

    fn four_d_to_2d<S>(&self, arr: ArrayBase<S, Ix4>) -> Result<ArrayBase<S, Ix2>>
    where
        S: Data<Elem = f32>,
    {
        let Reshape2 {
            channels,
            height,
            width,
        } = *self;
        let batch_size = arr.dim().0;

        let arr_size = arr.dim().2 * arr.dim().3;
        arr.into_shape_with_order((batch_size, channels * height * width))
            .map_err(|_| MlErr::SizeMismatch {
                what: "4d array to 2d array",
                got: arr_size,
                expected: self.height * self.width,
            })
    }
}
