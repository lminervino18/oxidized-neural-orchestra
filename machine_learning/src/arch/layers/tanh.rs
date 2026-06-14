use ndarray::{Array2, ArrayView2, ArrayViewMut2, azip};
use std::f32;

use crate::{Result, arch::layers::InplaceReshape};

#[derive(Clone, Debug, Default)]
pub struct Tanh {
    amp: f32,

    // Forward metadata
    activations: Array2<f32>,
}

impl Tanh {
    pub fn new(amp: f32) -> Self {
        let zeros = Array2::zeros((1, 1));

        Self {
            amp,
            activations: zeros,
        }
    }

    pub fn size(&self) -> usize {
        0
    }

    pub fn forward(&mut self, x: ArrayView2<f32>) -> Result<ArrayView2<'_, f32>> {
        self.activations.reshape_inplace(x.raw_dim());

        azip!((a in &mut self.activations, &x_in in &x) {
            let exp = x_in.exp();
            let neg_exp = (-x_in).exp();
            *a = self.amp * (exp - neg_exp) / (exp + neg_exp);
        });

        Ok(self.activations.view())
    }

    pub fn backward<'a>(
        &'a mut self,
        mut d: ArrayViewMut2<'a, f32>,
    ) -> Result<ArrayViewMut2<'a, f32>> {
        let one_over_amp = 1.0 / self.amp;

        azip!((d_in in &mut d, &a in &self.activations) {
            *d_in *= self.amp - (a.powi(2) * one_over_amp);
        });

        Ok(d)
    }
}
