use ndarray::{Array2, ArrayView2};

use super::LossFn;

pub struct Mse;

impl LossFn for Mse {
    fn loss(&self, y_pred: ArrayView2<f32>, y: ArrayView2<f32>) -> f32 {
        (&y_pred - &y)
            .mapv(|x| x.powi(2))
            .mean()
            .unwrap_or_default()
    }

    fn loss_prime(&self, y_pred: ArrayView2<f32>, y: ArrayView2<f32>) -> Array2<f32> {
        (&y_pred - &y) * (2.0 / y_pred.len() as f32)
    }
}
