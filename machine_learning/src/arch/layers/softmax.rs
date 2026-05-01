use std::f32;

use ndarray::{Array2, ArrayView2, ArrayViewMut2, azip, s};

use crate::{Result, arch::layers::InplaceReshape};

#[derive(Clone, Debug, Default)]
pub struct Softmax {
    // Forward metadata
    activations: Array2<f32>,
}

impl Softmax {
    pub fn new() -> Self {
        Self {
            activations: Array2::zeros((1, 1)),
        }
    }

    pub fn size(&self) -> usize {
        0
    }

    pub fn forward(&mut self, x: ArrayView2<f32>) -> Result<ArrayView2<'_, f32>> {
        self.activations.reshape_inplace(x.raw_dim());

        for b in 0..x.dim().0 {
            let mut activations_b = self.activations.slice_mut(s![b, ..]);
            let x_b = x.slice(s![b, ..]);

            let exp_row = x_b.mapv(|x| x.exp());
            let one_over_z = 1. / exp_row.sum();

            azip!((a in &mut activations_b, &x in &exp_row) {
                *a = x * one_over_z;
            });
        }

        Ok(self.activations.view())
    }

    pub fn backward<'a>(
        &'a mut self,
        mut d: ArrayViewMut2<'a, f32>,
    ) -> Result<ArrayViewMut2<'a, f32>> {
        for b in 0..d.dim().0 {
            let activations_b = self.activations.slice_mut(s![b, ..]);
            let d_b = d.slice_mut(s![b, ..]);

            let dot_product: f32 = activations_b
                .iter()
                .zip(d_b.iter())
                .map(|(&a, &d)| a * d)
                .sum();

            azip!((a in &activations_b, d in d_b) {
                *d = a * (*d - dot_product);
            });
        }

        Ok(d)
    }
}
