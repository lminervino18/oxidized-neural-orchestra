use ndarray::{Array2, ArrayView2, ArrayViewMut2};
use std::f32;

use crate::Result;

#[derive(Clone, Debug, Default)]
pub struct Sigmoid {
    amp: f32,

    // Forward metadata
    activations: Array2<f32>,

    delta: Array2<f32>, // TODO: sacar este buffer e ir pasando &mut Array2 para
                        // poder usar mapv_inplace y devolver esa ref u otra solución
}

impl Sigmoid {
    pub fn new(amp: f32) -> Self {
        let zeros = Array2::zeros((1, 1));

        Self {
            amp,
            activations: zeros.clone(),
            delta: zeros,
        }
    }

    fn sigmoid(&self, z: f32) -> f32 {
        1.0 / (1.0 + (-z).exp())
    }

    fn sigmoid_prime(&self, z: f32) -> f32 {
        let sigmoid = self.sigmoid(z);

        sigmoid * (1.0 - sigmoid)
    }

    pub fn forward(&mut self, x: ArrayView2<f32>) -> Result<ArrayView2<'_, f32>> {
        let sigmoid = |z| self.sigmoid(z);
        let amp = self.amp;

        self.activations = x.mapv(|z| amp * sigmoid(z));

        Ok(self.activations.view())
    }

    pub fn backward(&mut self, d: ArrayViewMut2<f32>) -> Result<ArrayViewMut2<'_, f32>> {
        let sigmoid_prime = |z| self.sigmoid_prime(z);
        let amp = self.amp;

        self.delta = d.mapv(|z| amp * sigmoid_prime(z));

        Ok(self.delta.view_mut())
    }

    pub fn size(&self) -> usize {
        0
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
