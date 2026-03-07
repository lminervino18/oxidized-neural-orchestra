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
        let amp = self.amp;

        azip!((d_in in &mut d, &a in &self.activations) {
            let s = a / amp;
            let local_grad = amp * s * (1.0 - s);
            *d_in *= local_grad;
        });

        Ok(d)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid_forward() {
        let amp = f32::consts::PI;
        let mut sigmoid = Sigmoid::new(amp);

        /* x =  [9 10 11]
         ***/
        let x = ArrayView2::from_shape((1, 3), &[9.0, 10.0, 11.0]).unwrap();

        /* sigmoid(x)
         ***/
        let expected = x.mapv(|z: f32| amp / (1.0 + (-z).exp()));

        let activations = sigmoid.forward(x).unwrap();

        assert_eq!(activations, expected);
    }

    #[test]
    fn test_sigmoid_backward() {
        let amp = f32::consts::PI;
        let mut sigmoid = Sigmoid::new(amp);

        /* x =  [9 10 11]
         ***/
        let mut d_raw = [9.0, 10.0, 11.0];
        let d = ArrayViewMut2::from_shape((1, 3), &mut d_raw).unwrap();

        /* sigmoid(x)
         ***/
        let s = |z: f32| 1.0 / (1.0 + (-z).exp());
        let expected = d.mapv(|z| amp * s(z) * (1.0 - s(z)));

        let activations = sigmoid.backward(d).unwrap();

        activations
            .iter()
            .zip(expected)
            .for_each(|(a, e)| assert!(f32::abs(a - e) < f32::EPSILON));
    }
}
