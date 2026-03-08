use ndarray::{Array2, ArrayView2, ArrayViewMut2, azip};
use std::f32;

use crate::{Result, arch::layers::InplaceReshape};

#[derive(Clone, Debug, Default)]
pub struct Sigmoid {
    amp: f32,

    // Forward metadata
    activations: Array2<f32>,
}

impl Sigmoid {
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
        self.activations.reshape_inplace((x.nrows(), x.ncols()));

        azip!((a in &mut self.activations, &x_in in &x) {
            *a = self.amp / (1.0 + (-x_in).exp());
        });

        Ok(self.activations.view())
    }

    pub fn backward<'a>(
        &'a mut self,
        mut d: ArrayViewMut2<'a, f32>,
    ) -> Result<ArrayViewMut2<'a, f32>> {
        azip!((d_in in &mut d, &a in &self.activations) {
            let s = a / self.amp;
            *d_in *= a * (1.0 - s);
        });

        Ok(d)
    }
}
