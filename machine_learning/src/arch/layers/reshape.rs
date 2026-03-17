use ndarray::{ArrayView2, ArrayView4, ArrayViewD, Ix2};

pub struct ReshapeMetadata {
    channels: usize,
    height: usize,
    width: usize,
}

pub enum Reshape {
    TwoDTo4D(ReshapeMetadata),
    FourDTo2D(ReshapeMetadata),
}

impl Reshape {
    pub fn forward<'a>(&self, input: ArrayViewD<'a, u32>) -> ArrayViewD<'a, u32> {
        match self {
            Self::TwoDTo4D(reshape) => {
                let input = input.into_dimensionality::<Ix2>().unwrap();

                self.two_d_to_4d(input, reshape).into_dyn()
            }
            Self::FourDTo2D(reshape) => {
                let input = input.into_dimensionality().unwrap();

                self.four_d_to_2d(input, reshape).into_dyn()
            }
        }
    }

    pub fn backward<'a>(&self, delta: ArrayViewD<'a, u32>) -> ArrayViewD<'a, u32> {
        match self {
            Self::TwoDTo4D(reshape) => {
                let delta = delta.into_dimensionality().unwrap();

                self.four_d_to_2d(delta, reshape).into_dyn()
            }
            Self::FourDTo2D(reshape) => {
                let delta = delta.into_dimensionality::<Ix2>().unwrap();

                self.two_d_to_4d(delta, reshape).into_dyn()
            }
        }
    }

    fn two_d_to_4d<'a>(
        &self,
        arr: ArrayView2<'a, u32>,
        reshape: &ReshapeMetadata,
    ) -> ArrayView4<'a, u32> {
        let ReshapeMetadata {
            channels,
            height,
            width,
        } = *reshape;
        let batch_size = arr.dim().0;

        arr.into_shape_with_order((batch_size, channels, height, width))
            .unwrap()
    }

    fn four_d_to_2d<'a>(
        &self,
        arr: ArrayView4<'a, u32>,
        reshape: &ReshapeMetadata,
    ) -> ArrayView2<'a, u32> {
        let ReshapeMetadata {
            channels,
            height,
            width,
        } = *reshape;
        let batch_size = arr.dim().0;

        arr.into_shape_with_order((batch_size, channels * height * width))
            .unwrap()
    }
}
