use comms::floats::Float01;
use ndarray::{Array2, ArrayView2, ArrayViewMut2, azip};
use std::f32;

use crate::{Result, arch::layers::InplaceReshape};

#[derive(Clone, Debug, Default)]
pub struct ReLU {
    slope: f32,
    activations: Array2<f32>,
}

impl ReLU {
    pub fn new(slope: Float01) -> Self {
        let zeros = Array2::zeros((1, 1));

        Self {
            slope: *slope,
            activations: zeros,
        }
    }

    pub fn size(&self) -> usize {
        0
    }

    pub fn forward(&mut self, x: ArrayView2<f32>) -> Result<ArrayView2<'_, f32>> {
        self.activations.reshape_inplace(x.raw_dim());

        azip!((a in &mut self.activations, &x_in in &x) {
            *a = x_in.max(self.slope * x_in);
        });

        Ok(self.activations.view())
    }

    pub fn backward<'a>(
        &'a mut self,
        mut d: ArrayViewMut2<'a, f32>,
    ) -> Result<ArrayViewMut2<'a, f32>> {
        azip!((d_in in &mut d, &a in &self.activations) {
            if a <= 0.0 {
                *d_in = self.slope * a;
            }
        });

        Ok(d)
    }
}
